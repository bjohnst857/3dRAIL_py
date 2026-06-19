"""3D RAIL IMEX advection-diffusion solver (Tucker format), orders 1/2/3.

Solves, with periodic boundary conditions on [-L/2, L/2]^3,

    u_t + (a1 u)_x + (a2 u)_y + (a3 u)_z = d1 u_xx + d2 u_yy + d3 u_zz + phi.

Diffusion is treated implicitly and advection + source explicitly.  The time
integrator is selectable: first-order IMEX111, second-order IMEX222, or
third-order IMEX443 (see imex.py).  The solution stays in Tucker low-rank format.

Command-line usage
------------------
    python main.py [--order {1,2,3}] [--test {1..8}] [--sweep {0,1}]

--order  IMEX order of accuracy: 1=IMEX111, 2=IMEX222, 3=IMEX443.  Default: 1.
--test   Which test case to run (1-8; see test_parameters.py).  Default: 1.
             1,2 -> constant-coefficient advection-diffusion (rank-1 / rank-2)
             3,6 -> rigid-body rotation with source (exact soln)
             4,5,7,8 -> rank tests (no exact solution)
--sweep  0 -> single lambda (CFL factor) value; 1 -> sweep lambda to check the
         order of accuracy.  Default: 0.
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

from test_parameters import TestParameters
from imex import imex_step, DiffMatrices
from helpers import tucker_full


def first_derivative_matrix(N, ds, L):
    """Spectral first-derivative matrix for periodic BCs [Trefethen, 2000].

    Skew-symmetric Toeplitz matrix (cot-based).  Used for the advection fluxes.
    """
    k = np.arange(1, N)
    col = np.zeros(N)
    col[1:] = 0.5 * (-1.0) ** k / np.tan(k * (2 * np.pi * ds / (2 * L)))
    # MATLAB's toeplitz(col, col([1 N:-1:2])): same first entry, then reversed tail.
    row = np.concatenate([[col[0]], col[N - 1:0:-1]])
    return (2 * np.pi / L) * toeplitz(col, row)


def second_derivative_matrix(N, ds, L):
    """Spectral second-derivative matrix for periodic BCs [Trefethen, 2000].

    Symmetric Toeplitz matrix.  Used for the (implicit) diffusion.  Not yet
    scaled by the diffusion coefficient.
    """
    first_row = np.zeros(N)
    first_row[0] = -1 / (3 * (2 * ds / L) ** 2) - 1 / 6
    first_row[1:] = 0.5 * (-1.0) ** np.arange(2, N + 1) / np.sin(
        (2 * np.pi * ds / L) * np.arange(1, N) / 2
    ) ** 2
    return (2 * np.pi / L) ** 2 * toeplitz(first_row)


def build_diff_matrices(p):
    """Assemble the DiffMatrices bundle from a TestParameters object ``p``."""
    Dx = first_derivative_matrix(p.Nx, p.dx, p.L)
    Dy = first_derivative_matrix(p.Ny, p.dy, p.L)
    Dz = first_derivative_matrix(p.Nz, p.dz, p.L)

    d1, d2, d3 = p.diffcoefs
    Dxx = d1 * second_derivative_matrix(p.Nx, p.dx, p.L)
    Dyy = d2 * second_derivative_matrix(p.Ny, p.dy, p.L)
    Dzz = d3 * second_derivative_matrix(p.Nz, p.dz, p.L)

    return DiffMatrices(first=[Dx, Dy, Dz], second=[Dxx, Dyy, Dzz])


def run(testnumber, lam, Tf, N, tol, order):
    """Run the solver for one lambda value.

    Returns (L1err, rankvals, tvals, p, U, G):
        L1err    : float       L^1 error vs the exact solution (NaN if none)
        rankvals : (Nt, 3)     multilinear rank [r1, r2, r3] at each time
        tvals    : (Nt,)       time levels
        p        : TestParameters
        U, G     : final Tucker solution (factor matrices and core)
    """
    p = TestParameters(testnumber, Tf, N, N, N)
    diff = build_diff_matrices(p)

    # Time step from the CFL-like condition; pin the last step exactly on Tf.
    cfl = p.CFLconstraints
    dt = lam / (cfl[0] / p.dx + cfl[1] / p.dy + cfl[2] / p.dz)
    tvals = np.arange(0, Tf, dt)
    if tvals[-1] != Tf:
        tvals = np.append(tvals, Tf)
    Nt = len(tvals)

    # Initial condition.
    U = [f.copy() for f in p.U]
    G = p.G.copy()
    MLR = list(G.shape)

    rankvals = np.zeros((Nt, 3), dtype=int)
    rankvals[0, :] = MLR

    for n in range(1, Nt):
        tn = tvals[n - 1]
        dtn = tvals[n] - tvals[n - 1]
        U, G, MLR = imex_step(order, U, G, MLR, p.A, p.B, p.C, p.P, tn, dtn, diff, tol)
        rankvals[n, :] = MLR

    # L1 error vs the exact solution (scaled by the cell measure).
    L1err = np.nan
    if p.compute_error == 1:
        u_approx = tucker_full(U, G)
        L1err = p.dx * p.dy * p.dz * np.sum(np.abs(u_approx - p.u_exact))

    return L1err, rankvals, tvals, p, U, G


def main():
    parser = argparse.ArgumentParser(
        description="3D RAIL IMEX advection-diffusion solver (orders 1, 2, 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--order", type=int, choices=[1, 2, 3], default=1,
                        help="IMEX order: 1=IMEX111, 2=IMEX222, 3=IMEX443. Default: 1.")
    parser.add_argument("--test", type=int, choices=list(range(1, 9)), default=1,
                        help="test case (1-8; see test_parameters.py). Default: 1.")
    parser.add_argument("--sweep", type=int, choices=[0, 1], default=0,
                        help="0=single lambda; 1=lambda sweep. Default: 0.")
    parser.add_argument("--Tf", type=float, default=10, help="final time. Default: 10.")
    parser.add_argument("--N", type=int, default=100, help="cells per dimension. Default: 100.")
    parser.add_argument("--tol", type=float, default=1e-4, help="truncation tolerance. Default: 1e-4.")
    args = parser.parse_args()

    if args.sweep == 0:
        Lambdavals = np.array([0.9])
    else:
        Lambdavals = np.arange(0.5, 2.01, 0.1)

    L1errvals = np.zeros(len(Lambdavals))
    rankvals = tvals = None
    t_start = time.perf_counter()                       # MATLAB's tic
    for k, lam in enumerate(Lambdavals):
        print(f"Starting lambda = {lam:.2f}")
        L1errvals[k], rankvals, tvals, p, U, G = run(
            args.test, lam, args.Tf, args.N, args.tol, args.order
        )
        print(f"  lambda={lam:.2f}, L1 error={L1errvals[k]:.3e}, "
              f"final ranks={[int(r) for r in rankvals[-1]]}")
    elapsed = time.perf_counter() - t_start             # MATLAB's toc
    print(f"Elapsed time is {elapsed:.6f} seconds.")

    # --- Plots ---
    fig1, ax1 = plt.subplots()
    ax1.plot(tvals, rankvals[:, 0], "b-", label="r1")
    ax1.plot(tvals, rankvals[:, 1], "g-", label="r2")
    ax1.plot(tvals, rankvals[:, 2], "m-", label="r3")
    ax1.plot(tvals, rankvals.sum(axis=1) / 3, "k-.", label="(r1+r2+r3)/3")
    ax1.set_xlabel("t"); ax1.set_ylabel("rank"); ax1.legend()
    ax1.set_title(f"Test {args.test}, IMEX order {args.order}: multilinear rank over time")
    ax1.set_ylim(bottom=0)
    fig1.savefig("figs/rank_vs_time.png", dpi=150, bbox_inches="tight")

    if args.sweep == 1:
        fig2, ax2 = plt.subplots()
        ax2.loglog(Lambdavals, L1errvals, "k-", label=f"N={args.N}")
        # reference line matching the selected order of accuracy
        ref = L1errvals[0] * (Lambdavals / Lambdavals[0]) ** args.order
        ax2.loglog(Lambdavals, ref, "b-.", label=f"order {args.order} ref")
        ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel(r"$L^1$ error")
        ax2.set_title(f"Test {args.test}, IMEX order {args.order}")
        ax2.legend()
        fig2.savefig("figs/l1_error.png", dpi=150, bbox_inches="tight")

    print("Saved plots: rank_vs_time.png" + (", l1_error.png" if args.sweep == 1 else ""))


if __name__ == "__main__":
    main()
