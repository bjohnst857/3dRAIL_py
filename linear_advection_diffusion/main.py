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
from lomac import build_lomac_data, macroscopic_quantities


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


def _rel_drift(q):
    """Final-time drift of a conserved quantity q(t) from its initial value.

    Returns the *relative* drift |q[-1]-q[0]|/|q[0]| when q[0] is nonzero, else
    the absolute drift |q[-1]-q[0]| (some quantities, e.g. mass of an odd initial
    profile, integrate to zero).
    """
    q0 = q[0]
    return abs(q[-1] - q0) / abs(q0) if abs(q0) > 1e-14 else abs(q[-1] - q0)


def run(testnumber, lam, Tf, N, tol, order, lomac_level=None):
    """Run the solver for one lambda value.

    Parameters
    ----------
    lomac_level : int or None
        None -> non-conservative HOSVD truncation; 0/1/2 -> conservative LoMaC
        preserving mass / mass+momentum / mass+momentum+energy.

    Returns (L1err, rankvals, tvals, p, U, G, macro):
        L1err    : float       L^1 error vs the exact solution (NaN if none)
        rankvals : (Nt, 3)     multilinear rank [r1, r2, r3] at each time
        tvals    : (Nt,)       time levels
        p        : TestParameters
        U, G     : final Tucker solution (factor matrices and core)
        macro    : dict of (Nt,) arrays  mass, Jx, Jy, Jz, energy over time
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

    # LoMaC data: weight function, constants, and the conserved reference moments
    # (those of the initial condition).  Always built so we can *track* the
    # macroscopic quantities; only *passed* to the step when LoMaC is requested.
    lomac_data = build_lomac_data(p, U, G, level=lomac_level if lomac_level is not None else 2)
    truncation = lomac_data if lomac_level is not None else None

    rankvals = np.zeros((Nt, 3), dtype=int)
    rankvals[0, :] = MLR

    # Physical quantities over time (mass, momentum x/y/z, energy).
    mass = np.zeros(Nt); Jx = np.zeros(Nt); Jy = np.zeros(Nt)
    Jz = np.zeros(Nt); energy = np.zeros(Nt)
    mass[0], Jx[0], Jy[0], Jz[0], energy[0] = macroscopic_quantities(U, G, lomac_data)

    for n in range(1, Nt):
        tn = tvals[n - 1]
        dtn = tvals[n] - tvals[n - 1]
        U, G, MLR = imex_step(order, U, G, MLR, p.A, p.B, p.C, p.P, tn, dtn,
                              diff, tol, truncation)
        rankvals[n, :] = MLR
        mass[n], Jx[n], Jy[n], Jz[n], energy[n] = macroscopic_quantities(U, G, lomac_data)

    # L1 error vs the exact solution (scaled by the cell measure).
    L1err = np.nan
    if p.compute_error == 1:
        u_approx = tucker_full(U, G)
        L1err = p.dx * p.dy * p.dz * np.sum(np.abs(u_approx - p.u_exact))

    macro = dict(mass=mass, Jx=Jx, Jy=Jy, Jz=Jz, energy=energy)
    return L1err, rankvals, tvals, p, U, G, macro


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
    parser.add_argument("--truncation",
                        choices=["none", "lomac0", "lomac01", "lomac012"], default="none",
                        help="truncation method: none=non-conservative HOSVD "
                             "(nonconstrun); lomac0=conserve mass; "
                             "lomac01=conserve mass+momentum; "
                             "lomac012=conserve mass+momentum+energy. Default: none.")
    args = parser.parse_args()
    # Map the choice to a LoMaC level (None = non-conservative HOSVD).
    lomac_level = {"none": None, "lomac0": 0, "lomac01": 1, "lomac012": 2}[args.truncation]

    if args.sweep == 0:
        Lambdavals = np.array([0.9])
    else:
        Lambdavals = np.arange(0.5, 2.01, 0.1)

    L1errvals = np.zeros(len(Lambdavals))
    rankvals = tvals = macro = None
    t_start = time.perf_counter()                       # MATLAB's tic
    for k, lam in enumerate(Lambdavals):
        print(f"Starting lambda = {lam:.2f}")
        L1errvals[k], rankvals, tvals, p, U, G, macro = run(
            args.test, lam, args.Tf, args.N, args.tol, args.order, lomac_level
        )
        # Conservation drift of mass and energy over the run (relative if nonzero).
        mass_drift = _rel_drift(macro["mass"])
        energy_drift = _rel_drift(macro["energy"])
        print(f"  lambda={lam:.2f}, L1 error={L1errvals[k]:.3e}, "
              f"final ranks={[int(r) for r in rankvals[-1]]}, "
              f"mass drift={mass_drift:.2e}, energy drift={energy_drift:.2e}")
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

    # Conservation of physical quantities over time (last lambda run), mirroring
    # the mass / momentum / energy figures of the MATLAB main.m.  Mass and energy
    # are shown as relative drift when their initial value is nonzero, else as
    # absolute drift (an odd initial profile integrates to zero).
    def drift_curve(ax, series, name):
        q0 = series[0]
        if abs(q0) > 1e-14:
            ax.semilogy(tvals, np.abs(series - q0) / abs(q0), "k-")
            ax.set_ylabel(f"relative {name}\n|q - q0|/|q0|")
        else:
            ax.semilogy(tvals, np.abs(series - q0), "k-")
            ax.set_ylabel(f"absolute {name}\n|q - q0|  (q0=0)")

    fig3, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    drift_curve(axes[0], macro["mass"], "mass")
    for comp, col in zip(["Jx", "Jy", "Jz"], ["k", "g", "m"]):
        axes[1].semilogy(tvals, np.abs(macro[comp] - macro[comp][0]), col + "-", label=comp)
    axes[1].set_ylabel("momentum |J - J0|"); axes[1].legend()
    drift_curve(axes[2], macro["energy"], "energy"); axes[2].set_xlabel("t")
    fig3.suptitle(f"Test {args.test}, order {args.order}, truncation={args.truncation}")
    fig3.savefig("figs/conservation.png", dpi=150, bbox_inches="tight")

    print("Saved plots: rank_vs_time.png, conservation.png"
          + (", l1_error.png" if args.sweep == 1 else ""))


if __name__ == "__main__":
    main()
