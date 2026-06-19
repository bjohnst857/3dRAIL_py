"""Rank test for the second-order (SDIRK2) diffusion solver.

Same test case as ``1st_order/rank_test.py``: a sum-of-two-Gaussians initial
condition advanced to a long final time on a coarse grid, tracking how the
multilinear rank evolves.  There is no exact solution to compare against --
the only output is the rank-vs-time plot.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as tl

from integrators import DiffMatrices, make_diff_matrix, dirk2

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
Lambdavals = np.array([0.5])   # single run

for k in range(len(Lambdavals)):
    lam = Lambdavals[k]
    print(f"Starting lambda = {lam}")
    N  = 10
    Nx = N;  Ny = N;  Nz = N
    L  = 14  # domain length in x, y, z

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
    Tf   = 50
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
    print("Parameters loaded")

    # ---------------------------------------------------------------------
    # Initial condition: sum of two Gaussians (no exact solution to compare)
    # meshgrid with indexing='ij' gives x-y-z ordering (matches MATLAB permute)
    # ---------------------------------------------------------------------
    x, y, z = np.meshgrid(xvals, yvals, zvals, indexing='ij')
    u = np.exp(-(y**2 + z**2)) * (np.exp(-(x-6)**2) + np.exp(-(x-8)**2))

    # Tucker decomposition of the initial condition
    tucker, _ = tl.mlsvd(u, tol=1e-6)
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

        print(f"Step {n}: r1={MLR[0]}, r2={MLR[1]}, r3={MLR[2]}, "
              f"R_kron={MLR[1]*MLR[2]}")

        if MLR[0] > 10 or MLR[1] > 10 or MLR[2] > 10:
            sys.exit()

        U, G, MLR = dirk2(U, G, MLR, diff, dtn, tol)
        rankvals[n, :] = MLR

# -------------------------------------------------------------------------
# Plot: multilinear rank over time
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots()
ax1.plot(tvals, rankvals[:, 0], 'b-',  linewidth=1.5, label='r1')
ax1.plot(tvals, rankvals[:, 1], 'g-',  linewidth=1.5, label='r2')
ax1.plot(tvals, rankvals[:, 2], 'm-',  linewidth=1.5, label='r3')
ax1.plot(tvals, rankvals.sum(axis=1)/3, 'k-.', linewidth=1.5, label='(r1+r2+r3)/3')
ax1.set_xlabel('t');  ax1.set_ylabel('rank')
ax1.set_xlim([0, Tf]); ax1.set_ylim([0, 11])
ax1.legend()

fig1.savefig('figs/rank_test/rank_vs_time.png', dpi=150, bbox_inches='tight')
