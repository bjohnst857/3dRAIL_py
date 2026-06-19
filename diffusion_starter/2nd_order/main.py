"""3D RAIL second-order (SDIRK2) diffusion solver (Tucker format).

Solves   u_t = d1*u_xx + d2*u_yy + d3*u_zz   on a periodic box, advancing the
low-rank Tucker solution with the two-stage DIRK2 step (``integrators.dirk2``),
the implicit half of the MATLAB IMEX222 scheme.

Command-line usage
------------------
    python main.py [--rank {1,2,3}] [--sweep {0,1}]

--rank   Initial-condition rank = number of sine modes summed in the IC, which
         sets the starting multilinear rank. Choices: 1, 2, 3.  Default: 1.
             1 -> sin(pi x) sin(pi y) sin(pi z)                  (rank 1)
             2 -> rank 1 + sin(2pi x) sin(2pi y) sin(2pi z)      (rank 2)
             3 -> rank 2 + sin(3pi x) sin(3pi y) sin(3pi z)      (rank 3)

--sweep  Lambda selection. Lambda is the CFL-like factor that sets the time step
         dt = lambda / (1/dx + 1/dy + 1/dz).  Choices: 0, 1.  Default: 0.
             0 -> single value, lambda = 0.5
             1 -> iterate over lambda = 0.1, 0.2, ..., 1.0

Examples
--------
    python main.py                      # rank-1 IC, single lambda (defaults)
    python main.py --rank 3             # rank-3 IC, single lambda
    python main.py --sweep 1            # rank-1 IC, lambda sweep (order check)
    python main.py --rank 2 --sweep 1   # rank-2 IC, lambda sweep
"""
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as tl

from integrators import DiffMatrices, make_diff_matrix, dirk2

# -------------------------------------------------------------------------
# Command-line arguments (all optional; see the module docstring above)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="3D RAIL second-order (SDIRK2) diffusion solver (Tucker format).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "examples:\n"
        "  python main.py                      rank-1 IC, single lambda (defaults)\n"
        "  python main.py --rank 3             rank-3 IC, single lambda\n"
        "  python main.py --sweep 1            rank-1 IC, lambda sweep (order check)\n"
        "  python main.py --rank 2 --sweep 1   rank-2 IC, lambda sweep\n"
    ),
)
parser.add_argument(
    "--rank", type=int, choices=[1, 2, 3], default=1,
    help="initial-condition rank = number of sine modes in the IC, setting the "
         "starting multilinear rank (1, 2, or 3). Default: 1.",
)
parser.add_argument(
    "--sweep", type=int, choices=[0, 1], default=0,
    help="lambda selection: 0 = single value (lambda=0.5); "
         "1 = iterate over lambda = 0.1, 0.2, ..., 1.0. Default: 0.",
)
args = parser.parse_args()
ic_rank = args.rank

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
if args.sweep == 0:
    Lambdavals = np.array([0.5])            # single lambda value
else:
    Lambdavals = np.arange(0.1, 1.1, 0.1)   # sweep over lambda

L1errvals = np.zeros((len(Lambdavals), 1))

for k in range(len(Lambdavals)):
    lam = Lambdavals[k]
    print(f"Starting lambda = {lam}")

    N  = 80
    Nx = N;  Ny = N;  Nz = N
    L  = 2  # domain length in x, y, z

    xvals = np.linspace(0, L, Nx + 1)
    yvals = np.linspace(0, L, Ny + 1)
    zvals = np.linspace(0, L, Nz + 1)
    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]
    dz = zvals[1] - zvals[0]
    xvals = xvals[:Nx] + dx / 2  # cell centers, Nx points
    yvals = yvals[:Ny] + dy / 2
    zvals = zvals[:Nz] + dz / 2

    dt   = lam / (1/dx + 1/dy + 1/dz)
    Tf   = 0.5
    tvals = np.arange(0, Tf, dt)
    if tvals[-1] != Tf:
        tvals = np.append(tvals, Tf)
    Nt = len(tvals)

    rankvals = np.zeros((Nt, 3), dtype=int)
    tol = 1.0e-6

    # ---------------------------------------------------------------------
    # Diffusion coefficients
    # ---------------------------------------------------------------------
    d1 = 1/3
    d2 = 1/4
    d3 = 1/5

    # ---------------------------------------------------------------------
    # Initial condition and exact solution
    # meshgrid with indexing='ij' gives x-y-z ordering (matches MATLAB permute)
    # The IC is a sum of `ic_rank` sine modes; mode m decays as
    #     exp(-(d1+d2+d3)*(m*pi)^2*t), giving a starting rank of ic_rank.
    # ---------------------------------------------------------------------
    x, y, z = np.meshgrid(xvals, yvals, zvals, indexing='ij')
    u       = np.zeros_like(x)
    u_exact = np.zeros_like(x)
    for m in range(1, ic_rank + 1):
        mode = np.sin(m*np.pi*x) * np.sin(m*np.pi*y) * np.sin(m*np.pi*z)
        u       += mode
        u_exact += np.exp(-(d1+d2+d3)*(m*np.pi)**2*Tf) * mode

    # Tucker decomposition of the initial condition (exact mlsvd, tol=1e-14)
    tucker, _ = tl.mlsvd(u, tol=1e-14, large_scale=False)
    U = list(tucker.factors)
    G = tucker.core

    print('Shape of core tensor:', np.shape(G))

    r1_n, r2_n, r3_n = G.shape
    MLR = [r1_n, r2_n, r3_n]
    rankvals[0, :] = MLR

    # ---------------------------------------------------------------------
    # Spectral differentiation matrices  [Trefethen, 2000]
    # ---------------------------------------------------------------------
    Dxx = d1 * make_diff_matrix(Nx, dx, L)
    Dyy = d2 * make_diff_matrix(Ny, dy, L)
    Dzz = d3 * make_diff_matrix(Nz, dz, L)
    diff = DiffMatrices(second=[Dxx, Dyy, Dzz])

    # ---------------------------------------------------------------------
    # Time loop (second-order DIRK2 step)
    # ---------------------------------------------------------------------
    for n in range(1, Nt):
        dtn = tvals[n] - tvals[n-1]

        if MLR[0] > 10 or MLR[1] > 10 or MLR[2] > 10:
            print(f"Rank blow-up at step {n}: MLR={MLR}")
            sys.exit()

        U, G, MLR = dirk2(U, G, MLR, diff, dtn, tol)
        rankvals[n, :] = MLR

    # L1 error scaled by domain measure
    u_approx = tl.tmprod(G, U, list(range(G.ndim)))
    L1errvals[k, 0] = (1/L**3) * dx*dy*dz * np.sum(np.abs(u_approx - u_exact))
    print(f"Lambda={lam:.2f}, L1 error={L1errvals[k, 0]:.2e}, Final ranks={MLR}")

# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------

# Multilinear rank over time
fig1, ax1 = plt.subplots()
ax1.plot(tvals, rankvals[:, 0], 'b-',  linewidth=1.5, label='r1')
ax1.plot(tvals, rankvals[:, 1], 'g-',  linewidth=1.5, label='r2')
ax1.plot(tvals, rankvals[:, 2], 'm-',  linewidth=1.5, label='r3')
ax1.plot(tvals, rankvals.sum(axis=1)/3, 'k-.', linewidth=1.5, label='(r1+r2+r3)/3')
ax1.set_xlabel('t');  ax1.set_ylabel('rank')
ax1.set_xlim([0, Tf]); ax1.set_ylim([0, 11])
ax1.legend()

# L1 error vs lambda, with an order-2 reference line
fig2, ax2 = plt.subplots()
ax2.loglog(Lambdavals, L1errvals, 'k-', linewidth=1.5, label=f'Nx=Ny=Nz={N}')
i0 = int(np.ceil(0.15 * len(Lambdavals)))
i1 = int(np.ceil(0.75 * len(Lambdavals)))
if i1 > i0:
    ax2.loglog(Lambdavals[i0:i1], 0.00005*Lambdavals[i0:i1]**2, 'k-.',
               linewidth=1.5, label='Order 2')
ax2.set_xlabel(r'$\lambda$');  ax2.set_ylabel(r'$L^1$ error')
ax2.legend(loc='upper left')

# Exact and numerical solution slices at z-index 49 (MATLAB index 50),
# plotted side by side with a shared color scale for direct comparison.
X_grid, Y_grid = np.meshgrid(xvals, yvals, indexing='ij')
u_approx_full = tl.tmprod(G, U, list(range(G.ndim)))
zk = 49
exact_slice  = u_exact[:, :, zk]
approx_slice = u_approx_full[:, :, zk]

vmin = min(exact_slice.min(), approx_slice.min())
vmax = max(exact_slice.max(), approx_slice.max())

fig3, (ax_e, ax_n) = plt.subplots(1, 2, figsize=(11, 4.5),
                                  sharex=True, sharey=True)
ax_e.pcolormesh(X_grid, Y_grid, exact_slice, shading='gouraud',
                vmin=vmin, vmax=vmax)
ax_e.set_title('Exact solution (z slice)')
ax_e.set_xlabel('x');  ax_e.set_ylabel('y')
ax_e.set_aspect('equal')

mesh_n = ax_n.pcolormesh(X_grid, Y_grid, approx_slice, shading='gouraud',
                         vmin=vmin, vmax=vmax)
ax_n.set_title('Numerical solution (z slice)')
ax_n.set_xlabel('x')
ax_n.set_aspect('equal')

fig3.suptitle(f'Rank-{ic_rank} ICs,  $\\lambda$={lam:.2f},  '
              f'Nx=Ny=Nz={N},  t={Tf}  (DIRK2)')
fig3.colorbar(mesh_n, ax=[ax_e, ax_n], label='u')

fig1.savefig('figs/rank_vs_time.png', dpi=150, bbox_inches='tight')
fig2.savefig('figs/l1_error.png', dpi=150, bbox_inches='tight')
fig3.savefig('figs/solution_comparison.png', dpi=150, bbox_inches='tight')
