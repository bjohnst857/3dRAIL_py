"""Small helper functions for the 3D RAIL advection-diffusion solver.

This single file collects the helper routines that live in separate files
in the original MATLAB code:

    MATLAB file          ->  function here
    -----------------------------------------
    TKR.m                ->  tkr
    tkron.m              ->  np.kron (called directly, no wrapper)
    red_aug_S.m          ->  red_aug_S
    nonconstrun.m        ->  nonconstrun
    tensoraddition.m     ->  tensoraddition
    (lmlragen builtin)   ->  tucker_full

Tucker-tensor convention used everywhere in this project
--------------------------------------------------------
A Tucker tensor is stored as a pair ``(factors, core)`` where

    factors = [U1, U2, U3]   list of factor matrices, shapes (N1,r1), (N2,r2), (N3,r3)
    core                     3D array, shape (r1, r2, r3)

and the full tensor is

    T[i,j,k] = sum_{a,b,c} core[a,b,c] * U1[i,a] * U2[j,b] * U3[k,c].
"""

import numpy as np
import pytensorlab as tl


def tucker_full(factors, core):
    """Rebuild the full dense tensor from a Tucker tensor.

    This is the MATLAB ``lmlragen({U1,U2,U3}, G)``.  pytensorlab has no
    ``lmlragen``, so we use ``tmprod`` (the n-mode product) instead.

    Parameters
    ----------
    factors : list of ndarray   [U1, U2, U3]
    core    : ndarray           shape (r1, r2, r3)

    Returns
    -------
    ndarray, shape (N1, N2, N3)
    """
    return tl.tmprod(core, factors, list(range(core.ndim)))


def tkr(A, B):
    """Transpose (row-wise) Khatri-Rao product of two matrices.

    Both inputs must have the same number of rows N.  Row k of the output is
    the Kronecker product of row k of A with row k of B:

        A : (N, r1)
        B : (N, r2)
        out[k, :] = kron(A[k, :], B[k, :])   ->  (N, r1*r2)

    Vectorised version of MATLAB's ``TKR.m`` (which loops over rows).
    """
    N, r1 = A.shape
    _, r2 = B.shape
    # einsum forms every pairwise product A[k,i]*B[k,j], then flatten (i,j).
    return np.einsum("ki,kj->kij", A, B).reshape(N, r1 * r2)


def flux_product(flow, solution):
    """Pointwise product of two Tucker tensors (stays in Tucker format).

    The flux terms in the PDE are products like ``E1 = a1(x,y,z) * u``, where
    both the flow field ``a1`` and the solution ``u`` are Tucker tensors.
    The product of two Tucker tensors is again Tucker:

        factor for each mode = tkr(flow_factor, solution_factor)
        core                 = np.kron(flow_core, solution_core)

    Parameters
    ----------
    flow     : (factors, core)   Tucker tensor for the flow field a_i
    solution : (factors, core)   Tucker tensor for u

    Returns
    -------
    (factors, core) : Tucker tensor for the product a_i * u
    """
    flow_factors, flow_core = flow
    sol_factors, sol_core = solution
    factors = [tkr(flow_factors[m], sol_factors[m]) for m in range(3)]
    core = np.kron(flow_core, sol_core)
    return factors, core


def tensoraddition(factors1, core1, factors2, core2):
    """Direct (block-diagonal) sum of two Tucker tensors -> represents their SUM.

    Port of ``tensoraddition.m``.  If T1 = (factors1, core1) and T2 =
    (factors2, core2), the result represents the full tensor ``T1 + T2`` while
    staying in Tucker format: each mode's factor is the horizontal stack of the
    two factors, and the new core places ``core1`` and ``core2`` in opposite
    corners of a larger, otherwise-zero core (so the two summands act on disjoint
    blocks of basis vectors).

    Parameters
    ----------
    factors1, factors2 : list[ndarray]   [U1, U2, U3] of each tensor
    core1, core2       : ndarray         3D cores, shapes (a1,b1,c1), (a2,b2,c2)

    Returns
    -------
    factors : list[ndarray]   [hstack(U1), hstack(U2), hstack(U3)]
    core    : ndarray         shape (a1+a2, b1+b2, c1+c2)
    """
    factors = [np.hstack([factors1[m], factors2[m]]) for m in range(3)]
    (a1, b1, c1), (a2, b2, c2) = core1.shape, core2.shape
    core = np.zeros((a1 + a2, b1 + b2, c1 + c2))
    core[:a1, :b1, :c1] = core1
    core[a1:, b1:, c1:] = core2
    return factors, core


def red_aug_K(Vx_dagger, Vx_RK, Vy_dagger, Vy_RK, Vz_dagger, Vz_RK):
    """Reduced augmentation before the K-steps (per-axis ranks).

    Used by the higher-order IMEX schemes (IMEX222 / IMEX443): the K-step bases
    ``V_star`` are obtained by augmenting the backward-Euler/IMEX111 predictor
    ``V_dagger`` with the bases from the previous RK stages (``V_RK``, typically
    ``np.hstack([V_{i-1}, ..., V_1, V_n])``), then truncating each axis
    independently at a 1e-12 singular-value threshold.  Port of ``red_aug_K.m``.

    Unlike :func:`red_aug_S`, the three axis ranks are kept independent (the
    K-step Sylvester solves do not require equal ranks).

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

    rx = int(np.sum(Sx > 1e-12))
    ry = int(np.sum(Sy > 1e-12))
    rz = int(np.sum(Sz > 1e-12))

    Vx_star = Qx @ Vx_t[:, :rx]
    Vy_star = Qy @ Vy_t[:, :ry]
    Vz_star = Qz @ Vz_t[:, :rz]
    return Vx_star, Vy_star, Vz_star, rx, ry, rz


def red_aug_S(Vx_ddagger, Vx_old, Vy_ddagger, Vy_old, Vz_ddagger, Vz_old):
    """Reduced augmentation before the S-step.

    Simoncini's solver needs the three reduced dimensions to be equal,
    r1 = r2 = r3 = R.  For each mode we stack the newly predicted basis
    (``V_ddagger`` from the K-step) next to the previous basis (``V_old``),
    orthonormalise, and keep the R most significant directions.

    Returns
    -------
    Vx_nn, Vy_nn, Vz_nn : ndarray   new orthonormal bases, each with R columns
    R : int                          the common reduced rank
    """
    Qx, Rx = np.linalg.qr(np.hstack([Vx_ddagger, Vx_old]), mode="reduced")
    Qy, Ry = np.linalg.qr(np.hstack([Vy_ddagger, Vy_old]), mode="reduced")
    Qz, Rz = np.linalg.qr(np.hstack([Vz_ddagger, Vz_old]), mode="reduced")

    Vx_t, Sx, _ = np.linalg.svd(Rx, full_matrices=False)
    Vy_t, Sy, _ = np.linalg.svd(Ry, full_matrices=False)
    Vz_t, Sz, _ = np.linalg.svd(Rz, full_matrices=False)

    # number of singular values above a small tolerance (numerical rank)
    rx = int(np.sum(Sx > 1e-12))
    ry = int(np.sum(Sy > 1e-12))
    rz = int(np.sum(Sz > 1e-12))

    R = min(max(rx, ry, rz), Rx.shape[1], Ry.shape[1], Rz.shape[1])

    Vx_nn = Qx @ Vx_t[:, :R]
    Vy_nn = Qy @ Vy_t[:, :R]
    Vz_nn = Qz @ Vz_t[:, :R]
    return Vx_nn, Vy_nn, Vz_nn, R


def nonconstrun(factors, core, tol):
    """Truncate a Tucker tensor with a sequentially-truncated HOSVD at ``tol``.

    This is the plain (non-conservative) truncation step.  We compute an
    ST-MLSVD of the core, fold its factor matrices into the existing bases, and
    return the smaller Tucker tensor plus its new multilinear rank.

    We do *not* use ``pytensorlab.mlsvd(core, tol=...)`` here: that routine's
    tolerance handling is buggy (``TruncationMLSVTolerance`` never advances its
    per-mode counter ``self._n``, so every mode is truncated at ``tol/ndim``
    instead of the intended ``tol/ndim, tol/(ndim-1), ..., tol/1``).  The later
    modes are under-truncated, which makes the multilinear rank creep upward
    every step and eventually blows up problems with a near-threshold
    singular-value cluster (e.g. test 3's rigid-body rotation).  See
    ``PYTENSORLAB_MLSVD_BUG.md``.  Instead we reproduce Tensorlab 3.0's exact
    ``mlsvd.m`` truncation rule (per-mode threshold ``tol/(N-n)`` for
    ``n = 0..N-1`` with accumulated relative error), matching the MATLAB code.

    Parameters
    ----------
    factors : list of ndarray   [Vx, Vy, Vz]
    core    : ndarray           current core
    tol     : float             relative truncation tolerance

    Returns
    -------
    new_factors : list of ndarray
    new_core    : ndarray
    rank        : list[int]   the new multilinear rank [r1, r2, r3]
    """
    S = np.asarray(core, dtype=float)
    N = S.ndim

    normsq = None       # squared Frobenius norm, taken from the first mode (Tensorlab T2)
    relerr = 0.0        # relative error accumulated from already-truncated modes
    mode_factors = []
    for n in range(N):
        # Mode-n unfolding (any column order gives the same left singular vectors
        # and singular values, so plain C-order reshape is fine here).
        Sn = np.moveaxis(S, n, 0).reshape(S.shape[n], -1)
        U, sv, _ = np.linalg.svd(Sn, full_matrices=False)
        if normsq is None:
            normsq = float(np.dot(sv, sv))

        # Largest number of trailing (smallest) singular values we may discard
        # while keeping the accumulated relative error below the per-mode budget.
        tail = np.cumsum(sv[::-1] ** 2)               # tail[i] = energy of i+1 smallest
        within = np.where(relerr + np.sqrt(tail / normsq) < tol / (N - n))[0]
        if within.size:
            ndrop = int(within.max()) + 1
            relerr += float(np.sqrt(tail[ndrop - 1] / normsq))
            keep = max(sv.size - ndrop, 1)
        else:
            keep = sv.size

        Uk = U[:, :keep]
        mode_factors.append(Uk)
        # Sequential truncation: project this mode out of the working core.
        rest = tuple(S.shape[j] for j in range(N) if j != n)
        S = np.moveaxis((Uk.T @ Sn).reshape((keep,) + rest), 0, n)

    new_factors = [factors[m] @ mode_factors[m] for m in range(N)]
    return new_factors, S, list(S.shape)
