import sys

import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as tl
from scipy.linalg import toeplitz, solve_sylvester
from simoncini import simoncini

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
Lambdavals = np.array([0.5])   # single run


# L1errvals  = np.zeros((len(Lambdavals), 1))

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

    # -------------------------------------------------------------------------
    # Diffusion coefficients
    # -------------------------------------------------------------------------
    d1 = 1/3
    d2 = 1/4
    d3 = 1/5
    print("Parameters loaded")
    # -------------------------------------------------------------------------
    # Initial condition and exact solution
    # meshgrid with indexing='ij' gives x-y-z ordering (matches MATLAB permute)
    # -------------------------------------------------------------------------
    x, y, z = np.meshgrid(xvals, yvals, zvals, indexing='ij')
    # -------------------------------------------------------------------------
    # Rank Test IC's, sum of two gaussians
    u = np.exp(-(y**2 + z**2))*(np.exp(-(x-6)**2) + np.exp(-(x-8)**2))
    # Tucker decomposition of initial condition
    # tl.mlsvd returns a Tucker tensor object with .core and .factors
    
    # Use large_scale=False to get exact mlsvd, analogous to MATLAB's mlsvd with tol=1e-14
    tucker, _ = tl.mlsvd(u, tol=1e-6)
    U = list(tucker.factors)
    G = tucker.core

    print('Shape of core tensor:', np.shape(G))

    r1_n, r2_n, r3_n = G.shape
    MLR = [r1_n, r2_n, r3_n]
    rankvals[0, :] = MLR

    # -------------------------------------------------------------------------
    # Spectral differentiation matrices  [Trefethen, 2000]
    # -------------------------------------------------------------------------
    def make_diff_matrix(N, ds, L):
        k         = np.arange(1, N)
        first_row = np.zeros(N)
        first_row[0]  = -1/(3*(2*ds/L)**2) - 1/6
        first_row[1:] = (0.5 * (-1)**np.arange(2, N+1)
                         / np.sin((2*np.pi*ds/L) * k / 2)**2)
        return (2*np.pi/L)**2 * toeplitz(first_row)

    Dxx = d1 * make_diff_matrix(Nx, dx, L)
    Dyy = d2 * make_diff_matrix(Ny, dy, L)
    Dzz = d3 * make_diff_matrix(Nz, dz, L)

    # -------------------------------------------------------------------------
    # K-step helper
    # -------------------------------------------------------------------------
    def compute_K_step(D_self, V_self_n, V_self_star,
                       D_a, V_a_n, V_a_star, r_a,
                       D_b, V_b_n, V_b_star, r_b,
                       S_n, mode, N_dim, dtn):

        A_sys  = np.eye(N_dim) - dtn * D_self

        kron_a = tl.kron((D_a @ V_a_star).T @ V_a_star, np.eye(r_a))
        kron_b = tl.kron(np.eye(r_b), (D_b @ V_b_star).T @ V_b_star)
        B_sys  = -dtn * (kron_a + kron_b)

        V_proj = tl.kron(V_a_n.T @ V_a_star, V_b_n.T @ V_b_star)
        Q_sys  = V_self_n @ tl.tens2mat(S_n, row=mode) @ V_proj

        return solve_sylvester(A_sys, B_sys, Q_sys)

    # -------------------------------------------------------------------------
    # Time loop
    # -------------------------------------------------------------------------
    for n in range(1, Nt):
        dtn = tvals[n] - tvals[n-1]

        Vx_n = U[0]
        Vy_n = U[1]
        Vz_n = U[2]
        S_n  = G

        Vx_star = Vx_n
        Vy_star = Vy_n
        Vz_star = Vz_n

        r1_n = MLR[0]
        r2_n = MLR[1]
        r3_n = MLR[2]
        print(f"Step {n}: r1={r1_n}, r2={r2_n}, r3={r3_n}, R_kron={r2_n*r3_n}")

        if r1_n > 10 or r2_n > 10 or r3_n > 10:
            sys.exit()

        # -- K steps ----------------------------------------------------------
        K1 = compute_K_step(Dxx, Vx_n, Vx_star,
                             Dzz, Vz_n, Vz_star, r2_n,
                             Dyy, Vy_n, Vy_star, r3_n,
                             S_n, mode=0, N_dim=Nx, dtn=dtn)

        K2 = compute_K_step(Dyy, Vy_n, Vy_star,
                             Dzz, Vz_n, Vz_star, r1_n,
                             Dxx, Vx_n, Vx_star, r3_n,
                             S_n, mode=1, N_dim=Ny, dtn=dtn)

        K3 = compute_K_step(Dzz, Vz_n, Vz_star,
                             Dyy, Vy_n, Vy_star, r1_n,
                             Dxx, Vx_n, Vx_star, r2_n,
                             S_n, mode=2, N_dim=Nz, dtn=dtn)

        Vx_ddagger, _ = np.linalg.qr(K1)
        Vy_ddagger, _ = np.linalg.qr(K2)
        Vz_ddagger, _ = np.linalg.qr(K3)

        # -- Reduced augmentation for S step ----------------------------------
        Qx, Rx = np.linalg.qr(np.hstack([Vx_ddagger, Vx_n]), mode='reduced')
        Qy, Ry = np.linalg.qr(np.hstack([Vy_ddagger, Vy_n]), mode='reduced')
        Qz, Rz = np.linalg.qr(np.hstack([Vz_ddagger, Vz_n]), mode='reduced')

        Vx_temp, Sx_temp, _ = np.linalg.svd(Rx, full_matrices=False)
        Vy_temp, Sy_temp, _ = np.linalg.svd(Ry, full_matrices=False)
        Vz_temp, Sz_temp, _ = np.linalg.svd(Rz, full_matrices=False)

        rx = np.where(Sx_temp > 1e-12)[0][-1] + 1
        ry = np.where(Sy_temp > 1e-12)[0][-1] + 1
        rz = np.where(Sz_temp > 1e-12)[0][-1] + 1

        # r1 = r2 = r3 = R after reduced augmentation
        R = min(max(rx, ry, rz), min(Rx.shape[1], Ry.shape[1], Rz.shape[1]))

        Vx_nn = Qx @ Vx_temp[:, :R]
        Vy_nn = Qy @ Vy_temp[:, :R]
        Vz_nn = Qz @ Vz_temp[:, :R]

        # -- S step (Simoncini solver) ----------------------------------------
        A1_s = -dtn * Vy_nn.T @ (Dyy @ Vy_nn)
        A2_s = np.eye(R) - dtn * Vz_nn.T @ (Dzz @ Vz_nn)
        A3_s = -dtn * Vx_nn.T @ (Dxx @ Vx_nn)
        M1_s = np.eye(R)
        M2_s = np.eye(R)
        H1_s = np.eye(R)
        H3_s = np.eye(R)

        B1 = Vx_nn.T @ Vx_n
        B2 = Vy_nn.T @ Vy_n
        B3 = Vz_nn.T @ Vz_n
        B  = tl.tmprod(S_n, [B1, B2, B3], list(range(S_n.ndim)))

        _, S_nn = simoncini(A1_s, A2_s, A3_s, M1_s, M2_s, H1_s, H3_s, B)

        # -- Truncation -------------------------------------------------------
        tucker_nn, _ = tl.mlsvd(S_nn, tol=tol)
        SU = list(tucker_nn.factors)
        SG = tucker_nn.core

        # pytensorlab's mlsvd always returns three correctly-shaped factors
        # (e.g. (Rr, 1) at rank 1), unlike MATLAB which squeezes singleton modes,
        # so no rank-collapse guard is needed here.

        Vx_nn = Vx_nn @ SU[0]
        Vy_nn = Vy_nn @ SU[1]
        Vz_nn = Vz_nn @ SU[2]
        S_nn  = SG

        r1_nn, r2_nn, r3_nn = S_nn.shape

        # -- Update solution --------------------------------------------------
        U   = [Vx_nn, Vy_nn, Vz_nn]
        G   = S_nn
        MLR = [r1_nn, r2_nn, r3_nn]
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
