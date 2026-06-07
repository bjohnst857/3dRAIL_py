import numpy as np
import matplotlib.pyplot as plt
import pytensorlab as tl

lam = 0.5

N  = 80
Nx = N;  Ny = N;  Nz = N
L  = 2                                  # domain length in x, y, z

xvals = np.linspace(0, L, Nx + 1)
yvals = np.linspace(0, L, Ny + 1)
zvals = np.linspace(0, L, Nz + 1)
dx = xvals[1] - xvals[0]
dy = yvals[1] - yvals[0]
dz = zvals[1] - zvals[0]
xvals = xvals[:Nx] + dx / 2            # cell centers, Nx points
yvals = yvals[:Ny] + dy / 2
zvals = zvals[:Nz] + dz / 2

dt   = lam / (1/dx + 1/dy + 1/dz)
Tf   = 2
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
u       = np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)
u_exact = (np.exp(-(d1+d2+d3)*np.pi**2*Tf)
            * np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z))

# Tucker decomposition of initial condition
# tl.mlsvd returns a Tucker tensor object with .core and .factors
tucker, _ = tl.mlsvd(u, tol=1e-14, large_scale = False)
U = list(tucker.factors)
G = tucker.core

print('Shape of core tensor:', np.shape(G))