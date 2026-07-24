"""IMEX time steps for the 3D RAIL advection-diffusion solver, orders 1-3.

PDE (periodic BCs):

    u_t + (a1 u)_x + (a2 u)_y + (a3 u)_z = d1 u_xx + d2 u_yy + d3 u_zz + phi

Diffusion is treated *implicitly* with a diagonally-implicit Runge-Kutta (DIRK)
tableau; the advection fluxes are treated *explicitly*; the known source ``phi``
is evaluated at the implicit stage nodes.  Everything stays in Tucker format.

This file ports the MATLAB files

    IMEX111.m   ->  imex111   first-order  (single diagonal-implicit stage)
    IMEX222.m   ->  imex222   second-order (ARS(2,2,2), gamma = 1-1/sqrt(2))
    IMEX443.m   ->  imex443   third-order  (ARS(4,4,3), diagonal 1/2)

and adds a small :func:`imex_step` dispatcher so ``main.py`` can pick the order.
All three orders share one Sylvester/Simoncini implementation (:func:`k_step`,
:func:`s_step`): IMEX111 is just that machinery run once with diagonal
coefficient 1 and star basis = the current basis, so it needs no separate code
path.  The higher-order schemes call ``imex111`` as a subroutine for their
predictors, exactly as the MATLAB does.
"""
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pytensorlab as tl
from numpy.linalg import LinAlgError
from scipy.linalg import schur
from scipy.linalg.lapack import get_lapack_funcs

from helpers import flux_product, red_aug_K, red_aug_S, tucker_full
from simoncini import simoncini
from lomac import truncate


@dataclass
class DiffMatrices:
    """Bundle of spatial differentiation matrices for one run.

    first[m]   : (N_m, N_m)  first-derivative matrix (advection), skew-symmetric
    second[m]  : (N_m, N_m)  second-derivative matrix already scaled by the
                             diffusion coefficient d_m (diffusion), symmetric
    """
    first: list   # [Dx, Dy, Dz]
    second: list  # [Dxx, Dyy, Dzz]  (already multiplied by d1, d2, d3)
    # Memoized real Schur factorizations of the K-step "A" matrix; see
    # schur_of_A.  Keyed by (mode, theta); not part of equality/repr.
    _schur_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def schur_of_A(self, m, theta):
        """Real Schur factorization (T, Z) of A = I - theta * second[m].

        Every K-step solves a Sylvester equation A K + K B = Q whose left-hand
        matrix is ``A = I - theta * second[m]``, with ``theta = (stage coeff)*dt``.
        Within a run this ``A`` is the *same* for almost every time step (``dt``
        and ``second[m]`` are fixed), so the Bartels-Stewart algorithm keeps
        recomputing the *identical* Schur decomposition of a large (N_m x N_m)
        matrix.  We compute it once per distinct ``(m, theta)`` and reuse it,
        which removes the dominant cost of the solver.

        The cache key rounds ``theta`` to 12 significant figures.  The step size
        ``dt`` is meant to be constant, but ``main.py`` forms each ``dtn`` as a
        difference of ``numpy.arange`` values, so consecutive steps differ in the
        last bit or two (~1e-16 relative).  Rounding collapses that floating-point
        noise to a single cache entry (the corresponding change in ``A`` is far
        below machine precision), while genuinely different ``theta`` values --
        distinct stage coefficients, or the final shortened step -- stay separate.

        Returns
        -------
        (T, Z) : tuple of ndarray
            Real Schur form with ``A = Z @ T @ Z.T`` and ``Z`` orthogonal.
        """
        key = (m, float(f"{theta:.12g}"))
        factorization = self._schur_cache.get(key)
        if factorization is None:
            N_m = self.second[m].shape[0]
            A = np.eye(N_m) - theta * self.second[m]
            factorization = schur(A, output="real")   # A = Z @ T @ Z.T
            self._schur_cache[key] = factorization
        return factorization


def solve_sylvester_cached(schur_a, b, q):
    """Solve the Sylvester equation ``A X + X B = Q`` reusing a Schur form of A.

    This is exactly ``scipy.linalg.solve_sylvester`` (the Bartels-Stewart
    algorithm), except the caller passes in the precomputed real Schur
    factorization of ``A`` instead of having it recomputed on every call.  That
    factorization is the expensive, reusable part (see
    :meth:`DiffMatrices.schur_of_A`); the rest -- a small Schur of ``B`` and a
    triangular solve -- is cheap and genuinely changes every step.

    Parameters
    ----------
    schur_a : (T, Z)        real Schur form of A from ``scipy.linalg.schur``
    b       : (n, n) ndarray   trailing matrix of the Sylvester equation
    q       : (N, n) ndarray   right-hand side

    Returns
    -------
    x : (N, n) ndarray   solution, identical to ``solve_sylvester(A, b, q)``.
    """
    r, u = schur_a
    s, v = schur(b.conj().T, output="real")
    f = u.conj().T @ q @ v
    trsyl, = get_lapack_funcs(("trsyl",), (r, s, f))
    y, scale, info = trsyl(r, s, f, tranb="C")
    if info < 0:
        raise LinAlgError(f"Illegal value encountered in the {-info} term")
    return u @ (scale * y) @ v.conj().T


# =========================================================================
# Building blocks shared by every stage
# =========================================================================
def k_rhs_piece(factors, core, m, V_star, diff_dir=None, diff_mats=None):
    """One additive piece of a K-step right-hand side (mode ``m``).

    Mirrors the MATLAB expression, e.g. for mode 0 (x):

        F1 * tens2mat(core, 1) * kron( F3' * Vz_star , F2' * Vy_star )

    Mode ``m`` is kept at full grid resolution; the other two modes are
    projected onto the star bases ``V_star``.  If ``diff_dir`` is given, the
    differentiation matrix ``diff_mats[diff_dir]`` is applied to that mode's
    factor first (this is how an advection flux or a diffusion term enters).

    Parameters
    ----------
    factors  : list[ndarray]   the three Tucker factors of the tensor
    core     : ndarray         its core
    m        : int             mode being updated (0=x, 1=y, 2=z)
    V_star   : list[ndarray]   star bases [Vx*, Vy*, Vz*] to project onto
    diff_dir : int or None     mode to differentiate, or None for no derivative
    diff_mats: list[ndarray]   differentiation matrices (diff.first or diff.second)

    Returns
    -------
    ndarray, shape (N_m, r_a * r_b)
    """
    projected_factors = []
    for j in range(3):
        f = factors[j]
        if diff_dir is not None and j == diff_dir:
            f = diff_mats[diff_dir] @ f          # apply the derivative
        if j == m:
            projected_factors.append(f)          # self mode: keep full
        else:
            projected_factors.append(V_star[j].T @ f)   # other modes: project
    small = tl.tmprod(core, projected_factors, [0, 1, 2])
    return tl.tens2mat(small, row=m)


def s_rhs_piece(factors, core, V_new, diff_dir=None, diff_mats=None):
    """One additive piece of the S-step right-hand-side tensor.

    Mirrors the MATLAB expression  lmlragen({Vx'*F1, Vy'*F2, Vz'*F3}, core),
    i.e. project every mode of the Tucker tensor onto the new bases ``V_new``,
    giving a dense R x R x R tensor.  If ``diff_dir`` is given, the
    differentiation matrix is applied to that mode's factor first.

    Returns
    -------
    ndarray, shape (R, R, R)
    """
    projected_factors = []
    for j in range(3):
        f = factors[j]
        if diff_dir is not None and j == diff_dir:
            f = diff_mats[diff_dir] @ f
        projected_factors.append(V_new[j].T @ f)
    return tucker_full(projected_factors, core)


def fluxes_at(flow_fields, sol):
    """The three advection fluxes E1=a1*u, E2=a2*u, E3=a3*u as Tucker tensors.

    ``flow_fields`` is the triple (A(t), B(t), C(t)) of flow components at one
    time, and ``sol`` is the solution Tucker tensor they multiply.  Each flux is
    a pointwise product of two Tucker tensors, which stays Tucker.  Port of the
    ``E1 = {{TKR(...),...}, tkron(...)}`` lines in the MATLAB.
    """
    A_t, B_t, C_t = flow_fields
    E1 = flux_product(A_t, sol)   # a1 * u
    E2 = flux_product(B_t, sol)   # a2 * u
    E3 = flux_product(C_t, sol)   # a3 * u
    return E1, E2, E3


# =========================================================================
# Generic K-step and S-step driven by a list of right-hand-side "terms"
# =========================================================================
#
# A "term" is a tuple   (coeff, kind, tensor, flux_dir)   describing one
# additive contribution to a stage right-hand side:
#
#   coeff    : float     scalar weight (already includes dt where appropriate)
#   kind     : str       'identity'  -> transport / source: no derivative
#                        'diffusion' -> d1 u_xx + d2 u_yy + d3 u_zz of a stage
#                                       (expands into three second-derivative
#                                        pieces, one per mode)
#                        'flux'      -> one advection flux d/dx_d (a_d u):
#                                       differentiate in direction flux_dir
#   tensor   : (factors, core)   the Tucker tensor the operator acts on
#   flux_dir : int       derivative direction for a 'flux' term (unused otherwise)
#
# This term list is exactly the set of "B = B + lmlragen(...)" lines in the
# MATLAB S-step, and the same terms also drive the K-steps.  Listing them once
# keeps the higher-order schemes readable while still matching the MATLAB.

def _term_pieces_k(term, m, V_star, diff):
    """Build (and weight) the K-step pieces of one term for mode ``m``."""
    coeff, kind, tensor, flux_dir = term
    factors, core = tensor
    if kind == "identity":
        piece = k_rhs_piece(factors, core, m, V_star)
    elif kind == "flux":
        piece = k_rhs_piece(factors, core, m, V_star, flux_dir, diff.first)
    elif kind == "diffusion":
        # d1 u_xx + d2 u_yy + d3 u_zz: one second-derivative piece per mode.
        piece = (k_rhs_piece(factors, core, m, V_star, 0, diff.second)
                 + k_rhs_piece(factors, core, m, V_star, 1, diff.second)
                 + k_rhs_piece(factors, core, m, V_star, 2, diff.second))
    else:
        raise ValueError(f"unknown term kind: {kind}")
    return coeff * piece


def _term_pieces_s(term, V_new, diff):
    """Build (and weight) the S-step pieces of one term."""
    coeff, kind, tensor, flux_dir = term
    factors, core = tensor
    if kind == "identity":
        piece = s_rhs_piece(factors, core, V_new)
    elif kind == "flux":
        piece = s_rhs_piece(factors, core, V_new, flux_dir, diff.first)
    elif kind == "diffusion":
        piece = (s_rhs_piece(factors, core, V_new, 0, diff.second)
                 + s_rhs_piece(factors, core, V_new, 1, diff.second)
                 + s_rhs_piece(factors, core, V_new, 2, diff.second))
    else:
        raise ValueError(f"unknown term kind: {kind}")
    return coeff * piece


def k_step(m, a_diag, dtn, diff, V_star, terms):
    """Solve the K_m Sylvester equation; return the new orthonormal basis.

    The left-hand side is the implicit diffusion of the *current* stage (the
    diagonal DIRK coefficient ``a_diag``); the right-hand side is the sum of the
    explicit ``terms``.  The diagonal coefficient is pulled out so every stage
    of every IMEX order -- including the single stage of IMEX111 -- can share
    this one routine.
    """
    second = diff.second

    # Implicit diffusion in the self mode -> A side of the Sylvester equation,
    # A = I - (a_diag*dtn) * second[m].  This matrix repeats across steps (and
    # across stages that share the diagonal coefficient), so we reuse its cached
    # Schur factorization keyed on the scalar theta = a_diag*dtn.
    schur_A = diff.schur_of_A(m, a_diag * dtn)

    # Implicit diffusion in the other two modes -> B side.  The two non-m modes
    # in ascending order: a is the outer index, b the inner one.
    a, b = sorted(j for j in range(3) if j != m)
    M_a = (second[a] @ V_star[a]).T @ V_star[a]
    M_b = (second[b] @ V_star[b]).T @ V_star[b]
    r_a, r_b = V_star[a].shape[1], V_star[b].shape[1]
    B_sys = -a_diag * dtn * (np.kron(M_a, np.eye(r_b)) + np.kron(np.eye(r_a), M_b))

    # Right-hand side: sum the explicit terms.
    Q = None
    for term in terms:
        piece = _term_pieces_k(term, m, V_star, diff)
        Q = piece if Q is None else Q + piece

    K = solve_sylvester_cached(schur_A, B_sys, Q)
    V_ddagger, _ = np.linalg.qr(K)
    return V_ddagger


def s_step(a_diag, dtn, diff, V_new, R, terms):
    """Solve the S-step (core update) with Simoncini's tensor-Sylvester solver.

    Shared by every IMEX order (including IMEX111's single stage), with the
    diagonal DIRK coefficient ``a_diag`` pulled out.
    """
    second = diff.second
    Vx, Vy, Vz = V_new
    A1 = -a_diag * dtn * Vy.T @ (second[1] @ Vy)
    A2 = np.eye(R) - a_diag * dtn * Vz.T @ (second[2] @ Vz)
    A3 = -a_diag * dtn * Vx.T @ (second[0] @ Vx)
    I_R = np.eye(R)

    # Right-hand-side tensor B: sum the explicit terms, projected to R x R x R.
    B = None
    for term in terms:
        piece = _term_pieces_s(term, V_new, diff)
        B = piece if B is None else B + piece

    _, S_nn = simoncini(A1, A2, A3, I_R, I_R, I_R, I_R, B)
    return S_nn


def implicit_stage(U_n, terms, a_diag, dtn, diff, tol, predictor, previous_U,
                   lomac_data=None):
    """Run one implicit DIRK stage (stages 2..s of IMEX222 / IMEX443).

    Steps, in the same order as the MATLAB:
      1. reduced-augment the IMEX111 ``predictor`` basis with the previous RK
         stage bases (red_aug_K) to get the star bases V_star;
      2. three K-steps (k_step) -> V_ddagger;
      3. reduced-augment again for the S-step (red_aug_S, common rank R);
      4. S-step (s_step) -> new core;
      5. truncate (plain HOSVD, or conservative LoMaC).

    Parameters
    ----------
    U_n        : list[ndarray]      factors at t^n (always part of the history)
    terms      : list[tuple]        stage right-hand-side terms (see above)
    a_diag     : float              DIRK diagonal implicit coefficient
    dtn        : float              full time step
    diff       : DiffMatrices       differentiation matrices
    tol        : float              truncation tolerance
    predictor  : list[ndarray]      IMEX111 predictor factors at this stage time
    previous_U : list[list]         previous stage factors [U_1, ..., U_{i-1}]
    lomac_data : LomacData or None  conservative truncation data, or None

    Returns
    -------
    U_nn, G_nn, MLR_nn : updated factors, core, multilinear rank
    """
    # Reduced augmentation before the K-steps.  The MATLAB stacks the bases as
    # [U_{i-1}, ..., U_1, U_n] (later stages first, then t^n).
    history = list(reversed(previous_U)) + [U_n]
    Vx_RK = np.hstack([U[0] for U in history])
    Vy_RK = np.hstack([U[1] for U in history])
    Vz_RK = np.hstack([U[2] for U in history])

    Vx_s, Vy_s, Vz_s, _, _, _ = red_aug_K(
        predictor[0], Vx_RK, predictor[1], Vy_RK, predictor[2], Vz_RK)
    V_star = [Vx_s, Vy_s, Vz_s]

    # K-steps: update each factor matrix.
    Vx_dd = k_step(0, a_diag, dtn, diff, V_star, terms)
    Vy_dd = k_step(1, a_diag, dtn, diff, V_star, terms)
    Vz_dd = k_step(2, a_diag, dtn, diff, V_star, terms)

    # Reduced augmentation before the S-step: make the three ranks equal to R.
    Vx_nn, Vy_nn, Vz_nn, R = red_aug_S(
        Vx_dd, Vx_RK, Vy_dd, Vy_RK, Vz_dd, Vz_RK)
    V_new = [Vx_nn, Vy_nn, Vz_nn]

    # S-step: update the core, then truncate.
    S_nn = s_step(a_diag, dtn, diff, V_new, R, terms)
    return truncate(V_new, S_nn, tol, lomac_data)


def flux_terms(coeff, flow_fields, sol):
    """The three advection-flux terms for solution ``sol`` at one time.

    ``coeff`` already includes the (signed) dt weight.  Returns a list of three
    'flux' terms, one per derivative direction.
    """
    E1, E2, E3 = fluxes_at(flow_fields, sol)
    return [
        (coeff, "flux", E1, 0),   # d/dx (a1 u)
        (coeff, "flux", E2, 1),   # d/dy (a2 u)
        (coeff, "flux", E3, 2),   # d/dz (a3 u)
    ]


# =========================================================================
# IMEX111  --  first-order, single diagonal-implicit stage
# =========================================================================
def imex111(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data=None):
    """Advance the Tucker solution one first-order IMEX step.

    Port of IMEX111.m.  A single implicit stage with diagonal coefficient 1 and
    star basis equal to the current basis U_n.

    Parameters
    ----------
    U_n   : list of ndarray   current factor matrices [Vx, Vy, Vz]
    G_n   : ndarray           current core tensor (r1, r2, r3)
    MLR_n : list[int]         current multilinear rank [r1, r2, r3] (unused: the
                               star basis U_n already carries these ranks)
    A,B,C : callable(t)       flow fields, each returns a Tucker tensor (factors, core)
    P     : callable(t)       source term, returns a Tucker tensor (factors, core)
    tn    : float             current time t^n
    dtn   : float             time step
    diff  : DiffMatrices      differentiation matrices
    tol   : float             truncation tolerance
    lomac_data : LomacData or None
                  if given, use conservative LoMaC truncation; else plain HOSVD

    Returns
    -------
    U_nn, G_nn, MLR_nn : updated factors, core, and multilinear rank
    """
    sol_n = (U_n, G_n)

    terms = [(1.0, "identity", sol_n, None)]
    terms += flux_terms(-dtn, (A(tn), B(tn), C(tn)), sol_n)
    terms += [(dtn, "identity", P(tn + dtn), None)]

    V_star = U_n
    Vx_dd = k_step(0, 1.0, dtn, diff, V_star, terms)
    Vy_dd = k_step(1, 1.0, dtn, diff, V_star, terms)
    Vz_dd = k_step(2, 1.0, dtn, diff, V_star, terms)

    Vx_nn, Vy_nn, Vz_nn, R = red_aug_S(
        Vx_dd, U_n[0], Vy_dd, U_n[1], Vz_dd, U_n[2])
    V_new = [Vx_nn, Vy_nn, Vz_nn]

    S_nn = s_step(1.0, dtn, diff, V_new, R, terms)
    return truncate(V_new, S_nn, tol, lomac_data)


# =========================================================================
# IMEX222  --  second-order, ARS(2,2,2),  gamma = 1 - 1/sqrt(2)
# =========================================================================
def imex222(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data=None):
    """Advance the Tucker solution one second-order IMEX222 step.

    Port of IMEX222.m.  Two stages: stage 1 is IMEX111 over gamma*dt; stage 2
    lands at t^{n+1}.
    """
    gamma = 1 - 1/sqrt(2)
    delta = 1 - 1/(2*gamma)
    sol_n = (U_n, G_n)

    # --- Stage 1: t^(1) = t^n + gamma*dt, just an IMEX111 sub-step ----------
    U_1, G_1, _ = imex111(U_n, G_n, MLR_n, A, B, C, P, tn, gamma*dtn, diff, tol, lomac_data)
    sol_1 = (U_1, G_1)
    t1 = tn + gamma*dtn

    # --- Stage 2: t^(2) = t^{n+1} (implicit node c2 = 1, diagonal gamma) ----
    # Right-hand-side terms, matching the "B = B + lmlragen(...)" lines.
    terms = [
        (1.0,               "identity",  sol_n,        None),  # transport u^n
        ((1-gamma)*dtn,     "diffusion", sol_1,        None),  # diffusion of u^(1)
        ((1-gamma)*dtn,     "identity",  P(t1),        None),  # source at node c1
        (gamma*dtn,         "identity",  P(tn+dtn),    None),  # source at node c2 (diag)
    ]
    terms += flux_terms(-delta*dtn,     (A(tn), B(tn), C(tn)), sol_n)  # flux at t^n
    terms += flux_terms(-(1-delta)*dtn, (A(t1), B(t1), C(t1)), sol_1)  # flux at t^(1)

    # Predictor: IMEX111 over the full step (we keep only its basis).
    U_dag, _, _ = imex111(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data)
    return implicit_stage(U_n, terms, gamma, dtn, diff, tol, U_dag, [U_1], lomac_data)


# =========================================================================
# IMEX443  --  third-order, ARS(4,4,3),  diagonal coefficient 1/2
# =========================================================================
def imex443(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data=None):
    """Advance the Tucker solution one third-order IMEX443 step.

    Port of IMEX443.m.  Four stages; the diagonal implicit coefficient is 1/2.
    """
    a = 0.5                                # DIRK diagonal coefficient
    c1, c2, c3, c4 = 0.5, 2/3, 0.5, 1.0    # implicit stage nodes
    sol_n = (U_n, G_n)

    def flow(t):
        return (A(t), B(t), C(t))

    # --- Stage 1: t^(1) = t^n + c1*dt, IMEX111 sub-step ---------------------
    U_1, G_1, _ = imex111(U_n, G_n, MLR_n, A, B, C, P, tn, c1*dtn, diff, tol, lomac_data)
    sol_1 = (U_1, G_1)
    t1 = tn + c1*dtn

    # --- Stage 2: node c2 ---------------------------------------------------
    terms = [
        (1.0,        "identity",  sol_n,            None),
        ((1/6)*dtn,  "diffusion", sol_1,            None),
        ((1/6)*dtn,  "identity",  P(tn + c1*dtn),   None),   # source node c1
        ((1/2)*dtn,  "identity",  P(tn + c2*dtn),   None),   # source node c2 (diag)
    ]
    terms += flux_terms(-(11/18)*dtn, flow(tn), sol_n)   # E_n at t^n
    terms += flux_terms(-(1/18)*dtn,  flow(t1), sol_1)   # E_1 at t^(1)
    U_dag, _, _ = imex111(U_n, G_n, MLR_n, A, B, C, P, tn, c2*dtn, diff, tol, lomac_data)
    U_2, G_2, _ = implicit_stage(U_n, terms, a, dtn, diff, tol, U_dag, [U_1], lomac_data)
    sol_2 = (U_2, G_2)
    t2 = tn + c2*dtn

    # --- Stage 3: node c3 ---------------------------------------------------
    terms = [
        (1.0,         "identity",  sol_n,           None),
        ((-1/2)*dtn,  "diffusion", sol_1,           None),
        (( 1/2)*dtn,  "diffusion", sol_2,           None),
        ((-1/2)*dtn,  "identity",  P(tn + c1*dtn),  None),
        (( 1/2)*dtn,  "identity",  P(tn + c2*dtn),  None),
        (( 1/2)*dtn,  "identity",  P(tn + c3*dtn),  None),  # diagonal node c3
    ]
    terms += flux_terms(-(5/6)*dtn, flow(tn), sol_n)
    terms += flux_terms(+(5/6)*dtn, flow(t1), sol_1)     # -(-5/6)
    terms += flux_terms(-(1/2)*dtn, flow(t2), sol_2)
    U_dag, _, _ = imex111(U_n, G_n, MLR_n, A, B, C, P, tn, c3*dtn, diff, tol, lomac_data)
    U_3, G_3, _ = implicit_stage(U_n, terms, a, dtn, diff, tol, U_dag, [U_1, U_2], lomac_data)
    sol_3 = (U_3, G_3)
    t3 = tn + c3*dtn

    # --- Stage 4: t^{n+1} (node c4 = 1) -------------------------------------
    terms = [
        (1.0,         "identity",  sol_n,           None),
        (( 3/2)*dtn,  "diffusion", sol_1,           None),
        ((-3/2)*dtn,  "diffusion", sol_2,           None),
        (( 1/2)*dtn,  "diffusion", sol_3,           None),
        (( 3/2)*dtn,  "identity",  P(tn + c1*dtn),  None),
        ((-3/2)*dtn,  "identity",  P(tn + c2*dtn),  None),
        (( 1/2)*dtn,  "identity",  P(tn + c3*dtn),  None),
        (( 1/2)*dtn,  "identity",  P(tn + c4*dtn),  None),  # diagonal node c4
    ]
    terms += flux_terms(-(1/4)*dtn, flow(tn), sol_n)
    terms += flux_terms(-(7/4)*dtn, flow(t1), sol_1)
    terms += flux_terms(-(3/4)*dtn, flow(t2), sol_2)
    terms += flux_terms(+(7/4)*dtn, flow(t3), sol_3)     # -(-7/4)
    U_dag, _, _ = imex111(U_n, G_n, MLR_n, A, B, C, P, tn, c4*dtn, diff, tol, lomac_data)
    return implicit_stage(U_n, terms, a, dtn, diff, tol, U_dag, [U_1, U_2, U_3], lomac_data)


# =========================================================================
# Dispatcher
# =========================================================================
def imex_step(order, U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data=None):
    """Advance one time step with the chosen IMEX order (1, 2, or 3).

    ``lomac_data`` selects the truncation: None -> plain HOSVD (``nonconstrun``);
    a :class:`lomac.LomacData` bundle -> conservative LoMaC truncation.
    """
    if order == 1:
        return imex111(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data)
    if order == 2:
        return imex222(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data)
    if order == 3:
        return imex443(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data)
    raise ValueError(f"order must be 1, 2, or 3 (got {order})")
