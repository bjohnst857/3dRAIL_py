# Author: J. Nakao (MATLAB original)
# Python conversion: B. Johnston
#
# Parameters for the linear advection-diffusion-reaction test case.
#
# PDE: u_t + div(<A,B,C>u) - beta*u = d1*u_xx + d2*u_yy + d3*u_zz + P
#
# NOTE (linear reaction term): ``beta`` (docs/3dRAIL_linear_reaction_term_1.md)
# is hardcoded per test case as ``self.beta`` -- main.py reads it off the
# TestParameters instance (``p.beta``) rather than taking it as a run()/CLI
# argument, since each test's exact solution is derived for one specific beta.
#
# Tucker tensor convention throughout:
#   A Tucker tensor T is represented as a tuple (U, G) where
#     U = [U1, U2, U3]  list of factor matrices (numpy arrays)
#     G                 core tensor (3D numpy array, shape (r1, r2, r3))
#   so that T[i,j,k] = sum_{a,b,c} G[a,b,c] * U1[i,a] * U2[j,b] * U3[k,c]

import numpy as np


class TestParameters:
    """
    Stores parameters for the linear advection-diffusion-reaction test case
    on the domain [-L/2, L/2]^3.

    Parameters
    ----------
    testnumber : int
        Which test case to load. Only 1 is defined.
    Tf : float
        Final time.
    Nx, Ny, Nz : int
        Number of cells in each spatial direction.

    Attributes
    ----------
    L : float
        Domain half-width; spatial domain is [-L/2, L/2].
    xvals, yvals, zvals : ndarray, shape (Nx,), (Ny,), (Nz,)
        Cell-center coordinates.
    dx, dy, dz : float
        Uniform cell widths.
    CFLconstraints : ndarray, shape (3,)
        Upper bounds on max|f_i'(u)| for the CFL condition.
    diffcoefs : ndarray, shape (3,)
        Diffusion coefficients [d1, d2, d3].
    beta : float
        Linear reaction coefficient (+beta*u on the RHS); positive -> growth,
        negative -> decay. Hardcoded per test case; default 0.0.
    U : list of ndarray
        Factor matrices of the Tucker decomposition of the initial condition.
        U = [U1, U2, U3] with shapes (Nx, r1), (Ny, r2), (Nz, r3).
    G : ndarray, shape (r1, r2, r3)
        Core tensor of the Tucker decomposition of the initial condition.
    u_exact : ndarray or 0
        Full-grid exact solution at time Tf (only set when compute_error==1).
    compute_error : int
        1 if an exact solution is available; 0 otherwise.
    A, B, C : callable (float -> tuple)
        Flow-field components as functions of time t.
        Each returns a Tucker tensor (U_list, G_core) representing the
        spatially varying coefficient, e.g. A(t) = ([a1,a2,a3], G_A).
    P : callable (float -> tuple)
        Source term as a function of time t.
        Returns a Tucker tensor ([P1, P2, P3], Q) where Q is the core.
    """

    def __init__(self, testnumber: int, Tf: float, Nx: int, Ny: int, Nz: int):
        self.testnumber = testnumber
        self.Tf = Tf
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.beta = 0.0  # default; individual test cases override as needed
        self._setup(testnumber, Tf, Nx, Ny, Nz)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_centers(L, N):
        """Return (vals, dval) for N cell centers on [-L/2, L/2]."""
        edges = np.linspace(-L / 2, L / 2, N + 1)
        d = edges[1] - edges[0]
        centers = edges[:N] + d / 2
        return centers, d

    # ------------------------------------------------------------------
    # Main setup dispatcher
    # ------------------------------------------------------------------

    def _setup(self, testnumber, Tf, Nx, Ny, Nz):
        if testnumber == 1:
            self._test1(Tf, Nx, Ny, Nz)
        else:
            raise ValueError(f"Unknown test number: {testnumber}")

    # ------------------------------------------------------------------
    # Test case
    # ------------------------------------------------------------------

    def _test1(self, Tf, Nx, Ny, Nz):
        """
        Constant-coefficient linear advection-diffusion-reaction:
          u_t + u_x + u_y + u_z - u = u_xx + u_yy + u_zz
        i.e. beta=1, d1=d2=d3=1, a1=a2=a3=1, zero source (self.beta=1.0 below;
        main.py reads it off this TestParameters instance).

        IC: sin(x)sin(y)sin(z) + sin(2x)sin(2y)sin(2z) + sin(3x)sin(3y)sin(3z)
            (Tucker rank 3, diagonal core)

        Exact solution:
          exp((1-3)t)  sin(x-t)  sin(y-t)  sin(z-t)
        + exp((1-12)t) sin(2x-2t) sin(2y-2t) sin(2z-2t)
        + exp((1-27)t) sin(3x-3t) sin(3y-3t) sin(3z-3t)
        """
        self.L = 2 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([1.0, 1.0, 1.0])
        d1 = d2 = d3 = 1.0
        self.diffcoefs = np.array([d1, d2, d3])
        self.beta = 1.0

        # Tucker IC: 3 components per mode (k=1,2,3), diagonal (3,3,3) core.
        self.U = [
            np.column_stack([np.sin(xvals), np.sin(2 * xvals), np.sin(3 * xvals)]),
            np.column_stack([np.sin(yvals), np.sin(2 * yvals), np.sin(3 * yvals)]),
            np.column_stack([np.sin(zvals), np.sin(2 * zvals), np.sin(3 * zvals)]),
        ]
        G = np.zeros((3, 3, 3))
        G[0, 0, 0] = 1.0
        G[1, 1, 1] = 1.0
        G[2, 2, 2] = 1.0
        self.G = G

        # Exact solution on full (Nx, Ny, Nz) grid  [indexing='ij' -> (x,y,z)]
        x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
        self.u_exact = np.zeros_like(x)
        for k in (1, 2, 3):
            lam_k = self.beta - 3 * d1 * k**2
            self.u_exact += (
                np.exp(lam_k * Tf)
                * np.sin(k * (x - Tf))
                * np.sin(k * (y - Tf))
                * np.sin(k * (z - Tf))
            )
        self.compute_error = 1

        # Flow fields: each is a Tucker tensor ({factors}, core)
        # A = 1 everywhere -> factors are all-ones vectors, scalar core [1]
        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        core1 = np.array([[[1.0]]])
        self.A = lambda t: ([ones_x, ones_y, ones_z], core1)
        self.B = lambda t: ([ones_x, ones_y, ones_z], core1)
        self.C = lambda t: ([ones_x, ones_y, ones_z], core1)

        # Source term P: zero Tucker tensor
        zero_core = np.array([[[0.0]]])
        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core,
        )
