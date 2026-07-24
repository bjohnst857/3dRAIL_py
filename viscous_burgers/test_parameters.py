# Author: J. Nakao (MATLAB original)
# Python conversion: B. Johnston
#
# Parameters for each numerical test (viscous Burgers' equation)
#
# PDE: u_t + (1/2)(u^2)_x + (1/2)(u^2)_y + (1/2)(u^2)_z
#           = d1*u_xx + d2*u_yy + d3*u_zz + P
#
# Tucker tensor convention throughout:
#   A Tucker tensor T is represented as a tuple (U, G) where
#     U = [U1, U2, U3]  list of factor matrices (numpy arrays)
#     G                 core tensor (3D numpy array, shape (r1, r2, r3))
#   so that T[i,j,k] = sum_{a,b,c} G[a,b,c] * U1[i,a] * U2[j,b] * U3[k,c]
#
# Note: For the Burgers equation the flow-field A, B, C are not used as
# independent inputs (the nonlinearity is handled internally); they are
# set to 0 as placeholders matching the MATLAB original.

import numpy as np


class TestParameters:
    """
    Stores all parameters for a given numerical test of the 3D viscous
    Burgers' equation on the domain [-L/2, L/2]^3.

    Parameters
    ----------
    testnumber : int
        Which test case to load (1–2).
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
    U : list of ndarray
        Factor matrices of the Tucker IC decomposition.
        U = [U1, U2, U3] with shapes (Nx, r1), (Ny, r2), (Nz, r3).
    G : ndarray, shape (r1, r2, r3)
        Core tensor of the Tucker IC decomposition.
    u_exact : ndarray or 0
        Full-grid exact solution at time Tf (only set when compute_error==1).
    compute_error : int
        1 if an exact solution is available; 0 otherwise.
    A, B, C : callable
        Placeholder flow fields (not used for Burgers); each returns 0.
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
        self._setup(testnumber, Tf, Nx, Ny, Nz)

    @staticmethod
    def _cell_centers(L, N):
        """Return (vals, dval) for N cell centers on [-L/2, L/2]."""
        edges = np.linspace(-L / 2, L / 2, N + 1)
        d = edges[1] - edges[0]
        centers = edges[:N] + d / 2
        return centers, d

    def _setup(self, testnumber, Tf, Nx, Ny, Nz):
        if testnumber == 1:
            self._test1(Tf, Nx, Ny, Nz)
        elif testnumber == 2:
            self._test2(Tf, Nx, Ny, Nz)
        else:
            raise ValueError(f"Unknown test number: {testnumber}")

    def _test1(self, Tf, Nx, Ny, Nz):
        """
        Viscous Burgers' equation — accuracy test.
        Multilinear Tucker rank (2,2,2) initial condition.
        Exact solution: exp(-3*d1*Tf) * sin(x+y+z)

        IC Tucker decomposition:
          U[i] has columns [sin, cos] for each direction.
          G = cat(3, [[-1,0],[0,1]], [[0,1],[1,0]])  shape (2,2,2)
          This encodes: u = -sin(x)sin(y)sin(z) + cos(x)cos(y)sin(z)
                          + sin(x)cos(y)cos(z) + cos(x)sin(y)cos(z)
                       = sin(x+y+z)  (via angle-addition identity)

        Source term P encodes the forcing from the nonlinear term:
          Q = (3/2)*exp(-6*d1*t)*G_IC   (same core scaled by time)
          P_i = [sin(2*vals), cos(2*vals)]  (double-angle factors)
        """
        self.L = 2 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([1.0, 1.0, 1.0])
        d1 = d2 = d3 = 1 / 2
        self.diffcoefs = np.array([d1, d2, d3])

        # Tucker IC: rank-(2,2,2)
        self.U = [
            np.column_stack([np.sin(xvals), np.cos(xvals)]),
            np.column_stack([np.sin(yvals), np.cos(yvals)]),
            np.column_stack([np.sin(zvals), np.cos(zvals)]),
        ]
        # MATLAB: cat(3, [-1,0;0,1], [0,1;1,0])  → shape (2,2,2)
        self.G = np.stack(
            [np.array([[-1.0, 0.0], [0.0, 1.0]]),
             np.array([[0.0, 1.0], [1.0, 0.0]])],
            axis=2,
        )

        x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
        self.u_exact = np.exp(-3 * d1 * Tf) * np.sin(x + y + z)
        self.compute_error = 1

        # Placeholders — not used for Burgers
        self.A = lambda t: 0
        self.B = lambda t: 0
        self.C = lambda t: 0

        # Source term: P = (3/2)*exp(-6*d1*t) * Tucker(same G as IC, double-angle factors)
        G_IC = self.G.copy()

        def P1(t):
            return np.column_stack([np.sin(2 * xvals), np.cos(2 * xvals)])

        def P2(t):
            return np.column_stack([np.sin(2 * yvals), np.cos(2 * yvals)])

        def P3(t):
            return np.column_stack([np.sin(2 * zvals), np.cos(2 * zvals)])

        self.P = lambda t: ([P1(t), P2(t), P3(t)], (3 / 2) * np.exp(-6 * d1 * t) * G_IC)

    def _test2(self, Tf, Nx, Ny, Nz):
        """
        Viscous Burgers' equation — rank test.
        Varies diffusion coefficient d; no exact solution.
        Same IC Tucker structure as test 1.
        """
        self.L = 2 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([1.0, 1.0, 1.0])
        d1 = d2 = d3 = 1 / 2
        self.diffcoefs = np.array([d1, d2, d3])

        self.U = [
            np.column_stack([np.sin(xvals), np.cos(xvals)]),
            np.column_stack([np.sin(yvals), np.cos(yvals)]),
            np.column_stack([np.sin(zvals), np.cos(zvals)]),
        ]
        self.G = np.stack(
            [np.array([[-1.0, 0.0], [0.0, 1.0]]),
             np.array([[0.0, 1.0], [1.0, 0.0]])],
            axis=2,
        )
        self.u_exact = 0
        self.compute_error = 0

        self.A = lambda t: 0
        self.B = lambda t: 0
        self.C = lambda t: 0

        zero_core = np.array([[[0.0]]])
        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core,
        )
