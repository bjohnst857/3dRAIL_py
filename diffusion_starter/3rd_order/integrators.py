"""Time integrators for the 3D RAIL pure-diffusion problem (Tucker format).

Solves   u_t = d1*u_xx + d2*u_yy + d3*u_zz   in Tucker (low-rank) form.

Two integrators are provided:

* ``backward_euler`` -- first-order implicit step.  This is the loop body of
  ``1st_order/main.py`` lifted into a reusable subroutine.
* ``dirk2``          -- second-order, two-stage SDIRK2 step (gamma = 1-1/sqrt(2)).
  It is the *implicit* half of the MATLAB ``IMEX222.m`` scheme with every
  advection/flux and source term removed (the diffusion starter has neither),
  and it calls ``backward_euler`` twice exactly as ``IMEX222`` calls ``IMEX111``.

Tucker convention (matching the rest of the project): a tensor is the pair
(factors, core) with factors ``[U1, U2, U3]`` (each ``(N_i, r_i)``) and core
``G`` of shape ``(r1, r2, r3)`` so that
``T[i,j,k] = sum_{abc} G[a,b,c] U1[i,a] U2[j,b] U3[k,c]``.
"""
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pytensorlab as tl
from scipy.linalg import toeplitz, solve_sylvester

from simoncini import simoncini


# -------------------------------------------------------------------------
# Parameter bundle
# -------------------------------------------------------------------------
@dataclass
class DiffMatrices:
    """Spatial second-derivative matrices for one run.

    Attributes
    ----------
    second : list[ndarray]
        ``[Dxx, Dyy, Dzz]`` -- dense second-derivative matrices of shape
        ``(Nx, Nx)``, ``(Ny, Ny)``, ``(Nz, Nz)``, already scaled by the
        diffusion coefficients ``d1, d2, d3``.
    """
    second: list


# -------------------------------------------------------------------------
# Spectral differentiation matrix  [Trefethen, 2000]
# -------------------------------------------------------------------------
def make_diff_matrix(N, ds, L):
    """Spectral second-derivative matrix on a periodic grid.

    Parameters
    ----------
    N  : int     number of grid points
    ds : float   grid spacing
    L  : float   domain length

    Returns
    -------
    ndarray, shape (N, N)
    """
    k         = np.arange(1, N)
    first_row = np.zeros(N)
    first_row[0]  = -1/(3*(2*ds/L)**2) - 1/6
    first_row[1:] = (0.5 * (-1)**np.arange(2, N+1)
                     / np.sin((2*np.pi*ds/L) * k / 2)**2)
    return (2*np.pi/L)**2 * toeplitz(first_row)


# -------------------------------------------------------------------------
# Reduced augmentation helpers  (ports of red_aug_K.m / red_aug_S.m)
# -------------------------------------------------------------------------
def red_aug_K(Vx_dagger, Vx_RK, Vy_dagger, Vy_RK, Vz_dagger, Vz_RK):
    """Reduced augmentation returning per-axis ranks.

    Builds an orthonormal basis spanning ``[V_dagger | V_RK]`` for each axis and
    truncates each axis independently at a 1e-12 singular-value threshold.
    Used to form the augmented bases V_star for the DIRK2 K steps.

    Parameters
    ----------
    V*_dagger : ndarray (N_i, *)   backward-Euler predictor basis for axis i
    V*_RK     : ndarray (N_i, *)   stacked bases from previous RK stages
                                   (e.g. ``np.hstack([V_1, V_n])``)

    Returns
    -------
    Vx_star, Vy_star, Vz_star : ndarray   augmented orthonormal bases
    rx, ry, rz                : int       per-axis ranks
    """
    Qx, Rx = np.linalg.qr(np.hstack([Vx_dagger, Vx_RK]), mode="reduced")
    Qy, Ry = np.linalg.qr(np.hstack([Vy_dagger, Vy_RK]), mode="reduced")
    Qz, Rz = np.linalg.qr(np.hstack([Vz_dagger, Vz_RK]), mode="reduced")

    Vx_t, Sx, _ = np.linalg.svd(Rx, full_matrices=False)
    Vy_t, Sy, _ = np.linalg.svd(Ry, full_matrices=False)
    Vz_t, Sz, _ = np.linalg.svd(Rz, full_matrices=False)

    rx = int(np.where(Sx > 1e-12)[0][-1] + 1)
    ry = int(np.where(Sy > 1e-12)[0][-1] + 1)
    rz = int(np.where(Sz > 1e-12)[0][-1] + 1)

    Vx_star = Qx @ Vx_t[:, :rx]
    Vy_star = Qy @ Vy_t[:, :ry]
    Vz_star = Qz @ Vz_t[:, :rz]
    return Vx_star, Vy_star, Vz_star, rx, ry, rz


def red_aug_S(Vx_ddagger, Vx_RK, Vy_ddagger, Vy_RK, Vz_ddagger, Vz_RK):
    """Reduced augmentation forcing a common rank R (for Simoncini's solver).

    Same construction as ``red_aug_K`` but collapses the three axis ranks to a
    single ``R`` so the S-step coefficient matrices are square. Identical to the
    inlined block in ``1st_order/main.py``.

    Returns
    -------
    Vx_nn, Vy_nn, Vz_nn : ndarray (N_i, R)
    R                   : int   common rank
    """
    Qx, Rx = np.linalg.qr(np.hstack([Vx_ddagger, Vx_RK]), mode="reduced")
    Qy, Ry = np.linalg.qr(np.hstack([Vy_ddagger, Vy_RK]), mode="reduced")
    Qz, Rz = np.linalg.qr(np.hstack([Vz_ddagger, Vz_RK]), mode="reduced")

    Vx_t, Sx, _ = np.linalg.svd(Rx, full_matrices=False)
    Vy_t, Sy, _ = np.linalg.svd(Ry, full_matrices=False)
    Vz_t, Sz, _ = np.linalg.svd(Rz, full_matrices=False)

    rx = int(np.where(Sx > 1e-12)[0][-1] + 1)
    ry = int(np.where(Sy > 1e-12)[0][-1] + 1)
    rz = int(np.where(Sz > 1e-12)[0][-1] + 1)

    R = min(max(rx, ry, rz), Rx.shape[1], Ry.shape[1], Rz.shape[1])

    Vx_nn = Qx @ Vx_t[:, :R]
    Vy_nn = Qy @ Vy_t[:, :R]
    Vz_nn = Qz @ Vz_t[:, :R]
    return Vx_nn, Vy_nn, Vz_nn, R


# -------------------------------------------------------------------------
# Backward-Euler K step (V_star = V_n)
# -------------------------------------------------------------------------
def _be_k_step(D_self, V_self_n,
               D_a, V_a_n, r_a,
               D_b, V_b_n, r_b,
               S_n, mode, N_dim, dtn):
    """One backward-Euler Sylvester solve for a single factor matrix.

    Solves  (I - dtn*D_self) K - dtn*K*(...) = V_self_n [S_n]_(mode) (...).
    Identical to ``compute_K_step`` in ``1st_order/main.py`` (here V_star = V_n).
    """
    A_sys  = np.eye(N_dim) - dtn * D_self

    kron_a = tl.kron((D_a @ V_a_n).T @ V_a_n, np.eye(r_a))
    kron_b = tl.kron(np.eye(r_b), (D_b @ V_b_n).T @ V_b_n)
    B_sys  = -dtn * (kron_a + kron_b)

    V_proj = tl.kron(V_a_n.T @ V_a_n, V_b_n.T @ V_b_n)
    Q_sys  = V_self_n @ tl.tens2mat(S_n, row=mode) @ V_proj

    return solve_sylvester(A_sys, B_sys, Q_sys)


def backward_euler(U_n, G_n, MLR_n, diff, dtn, tol):
    """Advance the Tucker solution one first-order backward-Euler step.

    Parameters
    ----------
    U_n   : list[ndarray]   current factors [Vx, Vy, Vz], shapes (N_i, r_i)
    G_n   : ndarray         current core, shape (r1, r2, r3)
    MLR_n : list[int]       current multilinear rank [r1, r2, r3]
    diff  : DiffMatrices    bundle of [Dxx, Dyy, Dzz]
    dtn   : float           time step
    tol   : float           truncation tolerance for the core mlsvd

    Returns
    -------
    U_nn  : list[ndarray]   updated factors
    G_nn  : ndarray         updated core
    MLR_nn: list[int]       updated multilinear rank
    """
    Dxx, Dyy, Dzz = diff.second
    Vx_n, Vy_n, Vz_n = U_n
    S_n = G_n
    r1_n, r2_n, r3_n = MLR_n
    Nx, Ny, Nz = Dxx.shape[0], Dyy.shape[0], Dzz.shape[0]

    # -- K steps (self / a / b assignment matches 1st_order compute_K_step) ---
    K1 = _be_k_step(Dxx, Vx_n, Dzz, Vz_n, r2_n, Dyy, Vy_n, r3_n, S_n, 0, Nx, dtn)
    K2 = _be_k_step(Dyy, Vy_n, Dzz, Vz_n, r1_n, Dxx, Vx_n, r3_n, S_n, 1, Ny, dtn)
    K3 = _be_k_step(Dzz, Vz_n, Dyy, Vy_n, r1_n, Dxx, Vx_n, r2_n, S_n, 2, Nz, dtn)

    Vx_ddagger, _ = np.linalg.qr(K1)
    Vy_ddagger, _ = np.linalg.qr(K2)
    Vz_ddagger, _ = np.linalg.qr(K3)

    # -- Reduced augmentation for the S step ---------------------------------
    Vx_nn, Vy_nn, Vz_nn, R = red_aug_S(Vx_ddagger, Vx_n,
                                       Vy_ddagger, Vy_n,
                                       Vz_ddagger, Vz_n)

    # -- S step (Simoncini solver) -------------------------------------------
    A1_s = -dtn * Vy_nn.T @ (Dyy @ Vy_nn)
    A2_s = np.eye(R) - dtn * Vz_nn.T @ (Dzz @ Vz_nn)
    A3_s = -dtn * Vx_nn.T @ (Dxx @ Vx_nn)
    I_R  = np.eye(R)

    B = tl.tmprod(S_n, [Vx_nn.T @ Vx_n, Vy_nn.T @ Vy_n, Vz_nn.T @ Vz_n], [0, 1, 2])
    _, S_nn = simoncini(A1_s, A2_s, A3_s, I_R, I_R, I_R, I_R, B)

    # -- Truncation ----------------------------------------------------------
    return _truncate(Vx_nn, Vy_nn, Vz_nn, S_nn, tol)


# -------------------------------------------------------------------------
# Generic DIRK stage machinery (shared by dirk2 and dirk3)
# -------------------------------------------------------------------------
# Per-mode axis assignment (self, a, b) used throughout the K steps:
#   mode 0 -> self=x, a=z, b=y;  mode 1 -> self=y, a=z, b=x;
#   mode 2 -> self=z, a=y, b=x.
_AXIS = {0: (0, 2, 1), 1: (1, 2, 0), 2: (2, 1, 0)}


def _dirk_k_step(mode, a_diag, dtn, diff_list, U_n, S_n, V_star, prev_stages):
    """One implicit DIRK Sylvester solve for a single factor matrix.

    Solves   (I - a_diag*dtn*D_self) K - a_diag*dtn*K*(...) = Q,
    where Q is the projection onto the augmented bases (V_star) of

        u^n + dtn * sum_j coeff_j * L(u^(j)).

    This is the diffusion-only K1/K2/K3 block shared by the IMEX222 / IMEX443
    schemes: ``a_diag`` is the DIRK diagonal coefficient and ``prev_stages`` is
    a list of ``(coeff_j, U_j, S_j)`` contributions from earlier RK stages.

    Parameters
    ----------
    mode        : int           0/1/2 -- which factor matrix to update
    a_diag      : float         DIRK diagonal implicit coefficient
    dtn         : float         time step
    diff_list   : list[ndarray] [Dxx, Dyy, Dzz]
    U_n, S_n    : current factors [Vx,Vy,Vz] and core (the u^n term)
    V_star      : list[ndarray] augmented bases [Vx_star, Vy_star, Vz_star]
    prev_stages : list[(float, list[ndarray], ndarray)]
                  ``(coeff_j, U_j, S_j)`` for each earlier stage j
    """
    s, a, b = _AXIS[mode]
    D_self, D_a, D_b = diff_list[s], diff_list[a], diff_list[b]
    V_self_star, V_a_star, V_b_star = V_star[s], V_star[a], V_star[b]
    ra_star = V_a_star.shape[1]
    rb_star = V_b_star.shape[1]
    N_dim = D_self.shape[0]

    A_sys  = np.eye(N_dim) - a_diag * dtn * D_self
    kron_a = tl.kron((D_a @ V_a_star).T @ V_a_star, np.eye(rb_star))
    kron_b = tl.kron(np.eye(ra_star), (D_b @ V_b_star).T @ V_b_star)
    B_sys  = -a_diag * dtn * (kron_a + kron_b)

    # u^n term: project the old solution onto the augmented bases
    Sn_mat = tl.tens2mat(S_n, row=mode)
    Q = U_n[s] @ Sn_mat @ tl.kron(U_n[a].T @ V_a_star, U_n[b].T @ V_b_star)

    # explicit stage contributions: coeff_j * dtn * L(u^(j)), one diffusion
    # operator applied per axis in turn
    for coeff, U_j, S_j in prev_stages:
        Vs, Va, Vb = U_j[s], U_j[a], U_j[b]
        Sj_mat = tl.tens2mat(S_j, row=mode)
        t_self = (D_self @ Vs) @ Sj_mat @ tl.kron(Va.T @ V_a_star, Vb.T @ V_b_star)
        t_a    = Vs @ Sj_mat @ tl.kron((D_a @ Va).T @ V_a_star, Vb.T @ V_b_star)
        t_b    = Vs @ Sj_mat @ tl.kron(Va.T @ V_a_star, (D_b @ Vb).T @ V_b_star)
        Q = Q + coeff * dtn * (t_self + t_a + t_b)

    return solve_sylvester(A_sys, B_sys, Q)


def _dirk_s_step(a_diag, dtn, diff_list, V_nn, R, U_n, S_n, prev_stages):
    """Implicit DIRK S step: solve the core via Simoncini's tensor solver.

    RHS is the projection onto the new bases of  u^n + dtn*sum_j coeff_j*L(u^(j)),
    using the same ``a_diag`` and ``prev_stages`` as the K steps.
    """
    Dxx, Dyy, Dzz = diff_list
    Vx_nn, Vy_nn, Vz_nn = V_nn
    Vx_n, Vy_n, Vz_n = U_n

    A1 = -a_diag * dtn * Vy_nn.T @ (Dyy @ Vy_nn)
    A2 = np.eye(R) - a_diag * dtn * Vz_nn.T @ (Dzz @ Vz_nn)
    A3 = -a_diag * dtn * Vx_nn.T @ (Dxx @ Vx_nn)
    I_R = np.eye(R)

    O = tl.tmprod(S_n, [Vx_nn.T @ Vx_n, Vy_nn.T @ Vy_n, Vz_nn.T @ Vz_n], [0, 1, 2])
    for coeff, U_j, S_j in prev_stages:
        Vxj, Vyj, Vzj = U_j
        cj = coeff * dtn
        O = O + cj * tl.tmprod(S_j, [Vx_nn.T @ (Dxx @ Vxj),
                                     Vy_nn.T @ Vyj, Vz_nn.T @ Vzj], [0, 1, 2])
        O = O + cj * tl.tmprod(S_j, [Vx_nn.T @ Vxj,
                                     Vy_nn.T @ (Dyy @ Vyj), Vz_nn.T @ Vzj], [0, 1, 2])
        O = O + cj * tl.tmprod(S_j, [Vx_nn.T @ Vxj,
                                     Vy_nn.T @ Vyj, Vz_nn.T @ (Dzz @ Vzj)], [0, 1, 2])

    _, S_nn = simoncini(A1, A2, A3, I_R, I_R, I_R, I_R, O)
    return S_nn


def _dirk_stage(U_n, G_n, MLR_n, diff, dtn, tol, a_diag, c_i, aug_U, prev_stages):
    """One implicit DIRK stage (stages 2..s).

    Mirrors a stage of IMEX222/IMEX443: a backward-Euler predictor to the stage
    time, a reduced-augmentation update of the bases *before* the K steps, the K
    steps, a reduced augmentation for the S step, the S step, and truncation.

    Parameters
    ----------
    a_diag      : float            DIRK diagonal implicit coefficient
    c_i         : float            stage node (predictor uses step c_i*dtn)
    aug_U       : list[list]       previous stage factors [U_1, ..., U_{i-1}]
                                   (ascending); the augmentation set is these
                                   plus U_n
    prev_stages : list[(float, list, ndarray)]   stage RHS contributions
    """
    # Backward-Euler predictor to the stage time t^n + c_i*dtn
    U_dag, _, _ = backward_euler(U_n, G_n, MLR_n, diff, c_i * dtn, tol)

    # Reduced augmentation before the K steps: span [U_{i-1}..U_1 | U_n]
    hist = list(reversed(aug_U)) + [U_n]
    Vx_RK = np.hstack([U[0] for U in hist])
    Vy_RK = np.hstack([U[1] for U in hist])
    Vz_RK = np.hstack([U[2] for U in hist])
    Vx_star, Vy_star, Vz_star, _, _, _ = red_aug_K(
        U_dag[0], Vx_RK, U_dag[1], Vy_RK, U_dag[2], Vz_RK)
    V_star = [Vx_star, Vy_star, Vz_star]

    # K steps
    K1 = _dirk_k_step(0, a_diag, dtn, diff.second, U_n, G_n, V_star, prev_stages)
    K2 = _dirk_k_step(1, a_diag, dtn, diff.second, U_n, G_n, V_star, prev_stages)
    K3 = _dirk_k_step(2, a_diag, dtn, diff.second, U_n, G_n, V_star, prev_stages)

    Vx_ddagger, _ = np.linalg.qr(K1)
    Vy_ddagger, _ = np.linalg.qr(K2)
    Vz_ddagger, _ = np.linalg.qr(K3)

    # Reduced augmentation for the S step (common rank R), same history
    Vx_nn, Vy_nn, Vz_nn, R = red_aug_S(
        Vx_ddagger, Vx_RK, Vy_ddagger, Vy_RK, Vz_ddagger, Vz_RK)

    # S step + truncation
    S_nn = _dirk_s_step(a_diag, dtn, diff.second, [Vx_nn, Vy_nn, Vz_nn], R,
                        U_n, G_n, prev_stages)
    return _truncate(Vx_nn, Vy_nn, Vz_nn, S_nn, tol)


def dirk2(U_n, G_n, MLR_n, diff, dtn, tol):
    """Advance the Tucker solution one second-order SDIRK2 step.

    Two-stage, stiffly-accurate DIRK2 (gamma = 1-1/sqrt(2)), the implicit half
    of MATLAB IMEX222.m.  Stage 1 is backward Euler with step gamma*dtn; stage 2
    is a generic implicit DIRK stage with one earlier-stage contribution.

    Parameters / Returns : same signature and meaning as ``backward_euler``.
    """
    gamma = 1 - 1/sqrt(2)

    # Stage 1 : backward Euler with step gamma*dtn
    U_1, G_1, _ = backward_euler(U_n, G_n, MLR_n, diff, gamma * dtn, tol)

    # Stage 2 (= t^{n+1}) : u^{n+1} - gamma*dt*L = u^n + (1-gamma)*dt*L(u^(1))
    return _dirk_stage(U_n, G_n, MLR_n, diff, dtn, tol,
                       a_diag=gamma, c_i=1.0,
                       aug_U=[U_1],
                       prev_stages=[(1 - gamma, U_1, G_1)])


def dirk3(U_n, G_n, MLR_n, diff, dtn, tol):
    """Advance the Tucker solution one third-order DIRK step.

    Four-stage, stiffly-accurate L-stable DIRK with diagonal coefficient 1/2
    (the ARS(4,4,3) implicit tableau), the implicit half of MATLAB IMEX443.m.
    Stage 1 is backward Euler with step dtn/2; stages 2-4 are generic implicit
    DIRK stages whose RHS accumulate  dt * sum_j a_ij * L(u^(j)).

    Implicit Butcher tableau (diffusion operator L):
        c = [1/2, 2/3, 1/2, 1]
        a_2 = [1/6];                 a_3 = [-1/2, 1/2];
        a_4 = [3/2, -3/2, 1/2];      b   = a_4 (stiffly accurate -> u^{n+1}=u^(4))

    Parameters / Returns : same signature and meaning as ``backward_euler``.
    """
    a = 0.5  # DIRK diagonal coefficient (same for every stage)

    # Stage 1 : backward Euler with step dtn/2  (c_1 = 1/2)
    U_1, G_1, _ = backward_euler(U_n, G_n, MLR_n, diff, 0.5 * dtn, tol)

    # Stage 2 : c_2 = 2/3
    U_2, G_2, _ = _dirk_stage(U_n, G_n, MLR_n, diff, dtn, tol,
                              a_diag=a, c_i=2/3,
                              aug_U=[U_1],
                              prev_stages=[(1/6, U_1, G_1)])

    # Stage 3 : c_3 = 1/2
    U_3, G_3, _ = _dirk_stage(U_n, G_n, MLR_n, diff, dtn, tol,
                              a_diag=a, c_i=1/2,
                              aug_U=[U_1, U_2],
                              prev_stages=[(-1/2, U_1, G_1), (1/2, U_2, G_2)])

    # Stage 4 (= t^{n+1}) : c_4 = 1
    return _dirk_stage(U_n, G_n, MLR_n, diff, dtn, tol,
                       a_diag=a, c_i=1.0,
                       aug_U=[U_1, U_2, U_3],
                       prev_stages=[(3/2, U_1, G_1), (-3/2, U_2, G_2),
                                    (1/2, U_3, G_3)])


# -------------------------------------------------------------------------
# Shared truncation (mlsvd of the core, folded back into the factors)
# -------------------------------------------------------------------------
def _truncate(Vx_nn, Vy_nn, Vz_nn, S_nn, tol):
    """Compress the core via mlsvd and fold the small factors into the bases."""
    tucker_nn, _ = tl.mlsvd(S_nn, tol=tol)
    SU = list(tucker_nn.factors)
    SG = tucker_nn.core

    Vx_nn = Vx_nn @ SU[0]
    Vy_nn = Vy_nn @ SU[1]
    Vz_nn = Vz_nn @ SU[2]

    U_nn   = [Vx_nn, Vy_nn, Vz_nn]
    MLR_nn = list(SG.shape)
    return U_nn, SG, MLR_nn
