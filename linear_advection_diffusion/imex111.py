"""One first-order IMEX (IMEX111) time step of the 3D RAIL solver.

The PDE is

    u_t + (a1 u)_x + (a2 u)_y + (a3 u)_z = d1 u_xx + d2 u_yy + d3 u_zz + phi

with periodic BCs.  Diffusion is handled *implicitly* and the advection
fluxes + source *explicitly* (this is what "IMEX" means).  The whole solution
is kept in Tucker format (factor matrices U = [Vx, Vy, Vz] and core G).

The step has three parts, exactly as in the MATLAB ``IMEX111.m``:
  1. K-steps   - update each factor matrix (one Sylvester solve per mode)
  2. reduced augmentation - make the three reduced ranks equal to R
  3. S-step    - update the core G (one Simoncini tensor-Sylvester solve)
  4. truncation - trim the new Tucker tensor back down with an HOSVD

A note on index ordering (the #1 source of bugs in this port)
-------------------------------------------------------------
pytensorlab's ``tens2mat(T, row=m)`` unfolds a tensor using NumPy's C-order,
which is the *opposite* of MATLAB's column-major ``tens2mat``.  To stay safe
we never hand-write a Kronecker product for the right-hand sides: we build a
small projected Tucker tensor with ``tmprod`` and matricize it with
``tens2mat``.  Those two pytensorlab functions are internally consistent, so
the column ordering always matches.  The only place we *do* write a kron is the
Sylvester ``B`` matrix, and there we use the verified rule: for mode ``m`` the
other two modes pair in ascending order (smaller index outer, larger inner).
"""

from dataclasses import dataclass, field

import numpy as np
import pytensorlab as tl
from numpy.linalg import LinAlgError
from scipy.linalg import schur
from scipy.linalg.lapack import get_lapack_funcs

from helpers import flux_product, red_aug_S, tucker_full
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


def _kstep_term(factors, core, m, V_star, deriv_dir, first):
    """Build one right-hand-side term of the K_m Sylvester equation.

    The term is the mode-``m`` unfolding of a Tucker tensor whose mode-``m``
    factor is kept full (so the result has N_m rows) and whose other two modes
    are projected onto the star bases ``V_star``.  If ``deriv_dir`` is not
    None, the first-derivative matrix is applied to that mode's factor first
    (this is how an advection flux enters the right-hand side).

    Returns an (N_m, r_a * r_b) matrix, where r_a, r_b are the star-basis
    ranks of the two non-``m`` modes.
    """
    g = []
    for j in range(3):
        f = factors[j]
        if deriv_dir is not None and j == deriv_dir:
            f = first[j] @ f                 # apply d/dx_j to this factor
        if j == m:
            g.append(f)                      # self mode: keep full
        else:
            g.append(V_star[j].T @ f)        # other modes: project onto star basis
    projected = tl.tmprod(core, g, [0, 1, 2])
    return tl.tens2mat(projected, row=m)


def _k_step(m, solution, fluxes, source, V_star, MLR, dtn, diff):
    """Solve the K_m Sylvester equation and return the new orthonormal basis.

    Parameters
    ----------
    m        : int                      which mode to update (0=x, 1=y, 2=z)
    solution : (factors, core)          current solution u^n in Tucker format
    fluxes   : list of (factors, core)  [E1, E2, E3] = [a1 u, a2 u, a3 u]
    source   : (factors, core)          source term phi at t^{n+1}
    V_star   : list of ndarray          star bases [Vx*, Vy*, Vz*]
    MLR      : list[int]                current multilinear rank [r1, r2, r3]
    dtn      : float                    time step
    diff     : DiffMatrices

    Returns the QR factor of the K-step solution (new candidate basis).
    """
    first, second = diff.first, diff.second

    # Implicit diffusion in the self mode -> A side of the Sylvester equation,
    # A = I - dtn * second[m].  This matrix is the same every step, so we reuse
    # its (cached) Schur factorization instead of rebuilding and re-decomposing.
    schur_A = diff.schur_of_A(m, dtn)

    # Implicit diffusion in the other two modes -> B side.  The two non-m modes
    # in ascending order: a is the outer index, b the inner one.
    a, b = sorted(j for j in range(3) if j != m)
    M_a = (second[a] @ V_star[a]).T @ V_star[a]   # (r_a, r_a)
    M_b = (second[b] @ V_star[b]).T @ V_star[b]   # (r_b, r_b)
    r_a, r_b = MLR[a], MLR[b]
    B_sys = -dtn * (np.kron(M_a, np.eye(r_b)) + np.kron(np.eye(r_a), M_b))

    # Right-hand side: transport (current u) - dt * advection fluxes + dt * source.
    sol_factors, sol_core = solution
    Q = _kstep_term(sol_factors, sol_core, m, V_star, None, first)
    for dir_, (e_factors, e_core) in enumerate(fluxes):
        Q = Q - dtn * _kstep_term(e_factors, e_core, m, V_star, dir_, first)
    p_factors, p_core = source
    Q = Q + dtn * _kstep_term(p_factors, p_core, m, V_star, None, first)

    K = solve_sylvester_cached(schur_A, B_sys, Q)
    V_ddagger, _ = np.linalg.qr(K)
    return V_ddagger


def _project_to_core(factors, core, V_new, deriv_dir, first):
    """Project a Tucker tensor onto new bases, returning a dense R x R x R core.

    Used to assemble the right-hand-side tensor of the S-step.  Each mode is
    projected onto the corresponding column of ``V_new``; if ``deriv_dir`` is
    not None the first-derivative matrix is applied to that mode first.
    """
    g = []
    for j in range(3):
        f = factors[j]
        if deriv_dir is not None and j == deriv_dir:
            f = first[j] @ f
        g.append(V_new[j].T @ f)
    return tucker_full(g, core)


def _s_step(solution, fluxes, source, V_new, R, dtn, diff):
    """Solve the S-step (core update) with Simoncini's tensor-Sylvester solver.

    Returns the new core tensor of shape (R, R, R).
    """
    first, second = diff.first, diff.second
    Vx, Vy, Vz = V_new

    # Coefficient matrices for Simoncini's solver (same structure as the
    # diffusion-only solver: implicit diffusion, symmetric matrices).
    A1 = -dtn * Vy.T @ (second[1] @ Vy)
    A2 = np.eye(R) - dtn * Vz.T @ (second[2] @ Vz)
    A3 = -dtn * Vx.T @ (second[0] @ Vx)
    M1 = M2 = H1 = H3 = np.eye(R)

    # Right-hand-side tensor B (all terms projected to R x R x R, then summed).
    sol_factors, sol_core = solution
    B = _project_to_core(sol_factors, sol_core, V_new, None, first)          # transport
    for dir_, (e_factors, e_core) in enumerate(fluxes):                       # -dt * fluxes
        B = B - dtn * _project_to_core(e_factors, e_core, V_new, dir_, first)
    p_factors, p_core = source                                               # +dt * source
    B = B + dtn * _project_to_core(p_factors, p_core, V_new, None, first)

    _, S_nn = simoncini(A1, A2, A3, M1, M2, H1, H3, B)
    return S_nn


def imex111(U_n, G_n, MLR_n, A, B, C, P, tn, dtn, diff, tol, lomac_data=None):
    """Advance the Tucker solution one first-order IMEX step.

    Parameters
    ----------
    U_n   : list of ndarray   current factor matrices [Vx, Vy, Vz]
    G_n   : ndarray           current core tensor (r1, r2, r3)
    MLR_n : list[int]         current multilinear rank [r1, r2, r3]
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
    solution = (U_n, G_n)

    # Flux terms E_i = a_i * u, evaluated at the current time t^n.  Each is a
    # pointwise product of two Tucker tensors, which stays Tucker.
    fluxes = [
        flux_product(A(tn), solution),
        flux_product(B(tn), solution),
        flux_product(C(tn), solution),
    ]
    # Source term, evaluated at t^{n+1} (explicit, but taken at the new time).
    source = P(tn + dtn)

    # Star bases for the K-steps.  In IMEX111 these are just the current bases.
    V_star = U_n

    # --- K-steps: update each factor matrix ---------------------------------
    Vx_ddagger = _k_step(0, solution, fluxes, source, V_star, MLR_n, dtn, diff)
    Vy_ddagger = _k_step(1, solution, fluxes, source, V_star, MLR_n, dtn, diff)
    Vz_ddagger = _k_step(2, solution, fluxes, source, V_star, MLR_n, dtn, diff)

    # --- Reduced augmentation: make the three reduced ranks equal to R ------
    Vx_nn, Vy_nn, Vz_nn, R = red_aug_S(
        Vx_ddagger, U_n[0], Vy_ddagger, U_n[1], Vz_ddagger, U_n[2]
    )
    V_new = [Vx_nn, Vy_nn, Vz_nn]

    # --- S-step: update the core --------------------------------------------
    S_nn = _s_step(solution, fluxes, source, V_new, R, dtn, diff)

    # --- Truncation (plain HOSVD, or conservative LoMaC) --------------------
    U_nn, G_nn, MLR_nn = truncate(V_new, S_nn, tol, lomac_data)
    return U_nn, G_nn, MLR_nn
