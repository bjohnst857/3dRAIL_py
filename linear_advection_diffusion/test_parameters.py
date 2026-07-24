# Author: J. Nakao (MATLAB original)
# Python conversion: B. Johnston
#
# Parameters for each numerical test (linear advection-diffusion models)
#
# PDE: u_t + div(<A,B,C>u) = d1*u_xx + d2*u_yy + d3*u_zz + P
#
# Tucker tensor convention throughout:
#   A Tucker tensor T is represented as a tuple (U, G) where
#     U = [U1, U2, U3]  list of factor matrices (numpy arrays)
#     G                 core tensor (3D numpy array, shape (r1, r2, r3))
#   so that T[i,j,k] = sum_{a,b,c} G[a,b,c] * U1[i,a] * U2[j,b] * U3[k,c]

import numpy as np


class TestParameters:
    """
    Stores all parameters for a given numerical test of the linear
    advection-diffusion equation on the domain [-L/2, L/2]^3.

    Parameters
    ----------
    testnumber : int
        Which test case to load (1–8).
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
        elif testnumber == 2:
            self._test2(Tf, Nx, Ny, Nz)
        elif testnumber == 3:
            self._test3(Tf, Nx, Ny, Nz)
        elif testnumber == 4:
            self._test4(Tf, Nx, Ny, Nz)
        elif testnumber == 5:
            self._test5(Tf, Nx, Ny, Nz)
        elif testnumber == 6:
            self._test6(Tf, Nx, Ny, Nz)
        elif testnumber == 7:
            self._test7(Tf, Nx, Ny, Nz)
        elif testnumber == 8:
            self._test8(Tf, Nx, Ny, Nz)
        else:
            raise ValueError(f"Unknown test number: {testnumber}")

    # ------------------------------------------------------------------
    # Individual test cases
    # ------------------------------------------------------------------

    def _test1(self, Tf, Nx, Ny, Nz):
        """
        Constant coefficient:  u_t + u_x + u_y + u_z = (1/6)(u_xx + u_yy + u_zz)
        Accuracy test (Tucker rank 1).
        Exact solution: exp(-3*d1*Tf) * sin(x-Tf)*sin(y-Tf)*sin(z-Tf)
        """
        self.L = 2 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([1.0, 1.0, 1.0])
        d1 = d2 = d3 = 1 / 6
        self.diffcoefs = np.array([d1, d2, d3])

        # Tucker IC: rank-1, G is 1×1×1
        self.U = [
            np.sin(xvals).reshape(-1, 1),
            np.sin(yvals).reshape(-1, 1),
            np.sin(zvals).reshape(-1, 1),
        ]
        self.G = np.array([[[1.0]]])

        # Exact solution on full (Nx, Ny, Nz) grid  [indexing='ij' → (x,y,z)]
        x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
        self.u_exact = (
            np.exp(-3 * d1 * Tf)
            * np.sin(x - Tf)
            * np.sin(y - Tf)
            * np.sin(z - Tf)
        )
        self.compute_error = 1

        # Flow fields: each is a Tucker tensor ({factors}, core)
        # A = 1 everywhere → factors are all-ones vectors, scalar core [1]
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

    def _test2(self, Tf, Nx, Ny, Nz):
        """
        Constant coefficient:  u_t + u_x + u_y + u_z = (1/6)(u_xx + u_yy + u_zz)
        Accuracy test (Tucker rank 2, multilinear rank (3,3,3)).
        Exact solution: 1 + exp(-3d1*Tf)*sin(x-Tf)sin(y-Tf)sin(z-Tf)
                          + exp(-12d1*Tf)*sin(2(x-Tf))sin(2(y-Tf))sin(2(z-Tf))
        """
        self.L = 2 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([1.0, 1.0, 1.0])
        d1 = d2 = d3 = 1 / 6
        self.diffcoefs = np.array([d1, d2, d3])

        # Tucker IC: 3 components per mode → rank-(3,3,3) core
        self.U = [
            np.column_stack([np.ones(Nx), np.sin(xvals), np.sin(2 * xvals)]),
            np.column_stack([np.ones(Ny), np.sin(yvals), np.sin(2 * yvals)]),
            np.column_stack([np.ones(Nz), np.sin(zvals), np.sin(2 * zvals)]),
        ]
        # MATLAB: cat(3, [1,0,0;0,0,0;0,0,0], [0,0,0;0,1,0;0,0,0], [0,0,0;0,0,0;0,0,1])
        # → 3×3×3 core, identity-like slices stacked along axis 2
        G = np.zeros((3, 3, 3))
        G[0, 0, 0] = 1.0
        G[1, 1, 1] = 1.0
        G[2, 2, 2] = 1.0
        self.G = G

        x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
        self.u_exact = (
            1
            + np.exp(-3 * d1 * Tf) * np.sin(x - Tf) * np.sin(y - Tf) * np.sin(z - Tf)
            + np.exp(-3 * d1 * 4 * Tf)
            * np.sin(2 * (x - Tf))
            * np.sin(2 * (y - Tf))
            * np.sin(2 * (z - Tf))
        )
        self.compute_error = 1

        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        core1 = np.array([[[1.0]]])
        self.A = lambda t: ([ones_x, ones_y, ones_z], core1)
        self.B = lambda t: ([ones_x, ones_y, ones_z], core1)
        self.C = lambda t: ([ones_x, ones_y, ones_z], core1)

        zero_core = np.array([[[0.0]]])
        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core,
        )

    def _test3(self, Tf, Nx, Ny, Nz):
        """
        Rigid-body rotation about <0,0,1> with diffusion:
          u_t - y*u_x + x*u_y = (1/3)(u_xx + u_yy + u_zz)
        Accuracy test.
        Exact solution: exp(-(x^2 + 2y^2 + 3z^2 + 3*d1*Tf))
        """
        self.L = 4 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([self.L / 2, self.L / 2, 0.0])
        d1 = d2 = d3 = 1 / 3
        self.diffcoefs = np.array([d1, d2, d3])

        self.U = [
            np.exp(-(xvals**2)).reshape(-1, 1),
            np.exp(-2 * (yvals**2)).reshape(-1, 1),
            np.exp(-3 * (zvals**2)).reshape(-1, 1),
        ]
        self.G = np.array([[[1.0]]])

        x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
        self.u_exact = np.exp(-(x**2 + 2 * y**2 + 3 * z**2 + 3 * d1 * Tf))
        self.compute_error = 1

        # A = (-y) in x-direction: Tucker factors and 2D-rank-1 core
        # A(t) = ({[1, -y, 1]}, [1]) meaning the spatial field is 1 * (-yvals) * 1
        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        core1 = np.array([[[1.0]]])
        zero_core = np.array([[[0.0]]])

        self.A = lambda t: ([ones_x, (-yvals).reshape(-1, 1), ones_z], core1)
        self.B = lambda t: ([xvals.reshape(-1, 1), ones_y, ones_z], core1)
        self.C = lambda t: ([np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))], zero_core)

        # Source term P (Tucker format with multi-rank core Q and multi-column factors)
        # MATLAB: Q = exp(-3*d1*t)*cat(3,[9d1,0,-16d1;0,-2,0;-4d1,0,0],[-36d1,0,0;0,0,0;0,0,0])
        #         P1 = [exp(-x^2), x*exp(-x^2), x^2*exp(-x^2)]   (Nx x 3)
        #         P2 = [exp(-2y^2), y*exp(-2y^2), y^2*exp(-2y^2)] (Ny x 3)
        #         P3 = [exp(-3z^2), z^2*exp(-3z^2)]               (Nz x 2)
        Q_base = np.stack(
            [
                np.array([[9 * d1, 0, -16 * d1], [0, -2, 0], [-4 * d1, 0, 0]]),
                np.array([[-36 * d1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            ],
            axis=2,
        )  # shape (3, 3, 2)

        def P1(t):
            return np.column_stack([
                np.exp(-(xvals**2)),
                xvals * np.exp(-(xvals**2)),
                (xvals**2) * np.exp(-(xvals**2)),
            ])

        def P2(t):
            return np.column_stack([
                np.exp(-2 * (yvals**2)),
                yvals * np.exp(-2 * (yvals**2)),
                (yvals**2) * np.exp(-2 * (yvals**2)),
            ])

        def P3(t):
            return np.column_stack([
                np.exp(-3 * (zvals**2)),
                (zvals**2) * np.exp(-3 * (zvals**2)),
            ])

        self.P = lambda t: ([P1(t), P2(t), P3(t)], np.exp(-3 * d1 * t) * Q_base)

    def _test4(self, Tf, Nx, Ny, Nz):
        """
        Rigid-body rotation about <0,0,1> with diffusion:
          u_t - 2y*u_x + 2x*u_y = (1/12)(u_xx + u_yy + u_zz)
        Rank test #1 (no exact solution needed).
        """
        self.L = 4 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([self.L, self.L, 0.0])
        d1 = d2 = d3 = 1 / 12
        self.diffcoefs = np.array([d1, d2, d3])

        self.U = [
            np.exp(-(xvals**2)).reshape(-1, 1),
            np.exp(-9 * (yvals**2)).reshape(-1, 1),
            np.exp(-(zvals**2)).reshape(-1, 1),
        ]
        self.G = np.array([[[1.0]]])
        self.u_exact = 0
        self.compute_error = 0

        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        core2 = np.array([[[2.0]]])
        zero_core = np.array([[[0.0]]])

        self.A = lambda t: ([ones_x, (-yvals).reshape(-1, 1), ones_z], core2)
        self.B = lambda t: ([xvals.reshape(-1, 1), ones_y, ones_z], core2)
        self.C = lambda t: ([np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))], zero_core)

        zero_core1 = np.array([[[0.0]]])
        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core1,
        )

    def _test5(self, Tf, Nx, Ny, Nz):
        """
        Rigid-body rotation about <1,1,1> with diffusion:
          u_t + (-y+z)*u_x + (x-z)*u_y + (-x+y)*u_z = (1/12)(u_xx + u_yy + u_zz)
        Rank test #2 (no exact solution needed).

        The flow field is rank-2 in Tucker format:
          A(t) = 1*(-y) + 1*(z)  = Tucker({[1,-y],[1,z],[1,1]}, G_A)
        """
        self.L = 4 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([self.L, self.L, self.L])
        d1 = d2 = d3 = 1 / 12
        self.diffcoefs = np.array([d1, d2, d3])

        self.U = [
            np.exp(-2 * (xvals - np.pi / 2) ** 2).reshape(-1, 1),
            np.exp(-2 * (yvals + np.pi / 2) ** 2).reshape(-1, 1),
            np.exp(-2 * (zvals**2)).reshape(-1, 1),
        ]
        self.G = np.array([[[1.0]]])
        self.u_exact = 0
        self.compute_error = 0

        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))

        # MATLAB: A = {{[1,1], [-y,y_'], [1,z_']}, cat(3,[0,-1],[1,0])}
        # i.e. factors have 2 columns, core is 1×2×2
        # → A(x,y,z) = -y*1 + z*1  (rank-2 Tucker)
        G_A = np.stack([np.array([[0, -1]]), np.array([[1, 0]])], axis=2)  # shape (1,2,2)
        G_B = np.stack([np.array([[0], [1]]), np.array([[-1], [0]])], axis=2)  # shape (2,1,2)
        G_C = np.array([[[0, 1], [-1, 0]]]).reshape(1, 2, 2)  # shape (1,2,2) ... wait

        # Let me re-derive from MATLAB:
        # A = {{ones(Nx,1), [ones(Ny,1),yvals'], [ones(Nz,1),zvals']}, cat(3,[0,-1],[1,0])}
        #   factors: Ax=(Nx,1), Ay=(Ny,2), Az=(Nz,2)
        #   core:   cat(3,[0,-1],[1,0]) = stack([[0,-1],[1,0]]) along dim3 → (1,2,2)
        #   → A(y,z) = core[0,i,j]*Ay[:,i]*Az[:,j] summed over i,j
        #            = 0*1*1 + (-1)*1*z + 1*y*1 + 0*y*z  = -z + y  ← matches -y+z in the PDE (roles differ)
        # B = {{[ones,x], ones(Ny), [ones,z]}, cat(3,[0;1],[-1;0])} → core (2,1,2)
        #   → B(x,z) = 0*1*1 + (-1)*1*z + 1*x*1 + 0*x*z = x - z ✓
        # C = {{[ones,x],[ones,y],ones}, [0 1;-1 0]} → core (2,2,1)
        #   → C(x,y) = 0*1*1 + 1*1*y + (-1)*x*1 + 0*x*y = y - x ✓

        Ax = ones_x                                                          # (Nx, 1)
        Ay = np.column_stack([np.ones(Ny), yvals])                          # (Ny, 2)
        Az = np.column_stack([np.ones(Nz), zvals])                          # (Nz, 2)
        core_A = np.stack([np.array([[0, -1]]), np.array([[1, 0]])], axis=2) # (1, 2, 2)
        self.A = lambda t: ([Ax, Ay, Az], core_A)

        Bx = np.column_stack([np.ones(Nx), xvals])                          # (Nx, 2)
        By = ones_y                                                          # (Ny, 1)
        Bz = np.column_stack([np.ones(Nz), zvals])                          # (Nz, 2)
        core_B = np.stack([np.array([[0], [1]]), np.array([[-1], [0]])], axis=2) # (2, 1, 2)
        self.B = lambda t: ([Bx, By, Bz], core_B)

        Cx = np.column_stack([np.ones(Nx), xvals])                          # (Nx, 2)
        Cy = np.column_stack([np.ones(Ny), yvals])                          # (Ny, 2)
        Cz = ones_z                                                          # (Nz, 1)
        core_C = np.array([[[0, 1], [-1, 0]]]).reshape(2, 2, 1)             # (2, 2, 1)
        self.C = lambda t: ([Cx, Cy, Cz], core_C)

        zero_core = np.array([[[0.0]]])
        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core,
        )

    def _test6(self, Tf, Nx, Ny, Nz):
        """
        Rigid-body rotation about <0,0,1> with time-dependent flow:
          u_t - t*y*u_x + t*x*u_y = (1/3)(u_xx + u_yy + u_zz)
        Accuracy test.
        Exact solution: exp(-(x^2 + 2y^2 + 3z^2 + 3*d1*Tf))
        """
        self.L = 4 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([Tf * self.L / 2, Tf * self.L / 2, 0.0])
        d1 = d2 = d3 = 1 / 3
        self.diffcoefs = np.array([d1, d2, d3])

        self.U = [
            np.exp(-(xvals**2)).reshape(-1, 1),
            np.exp(-2 * (yvals**2)).reshape(-1, 1),
            np.exp(-3 * (zvals**2)).reshape(-1, 1),
        ]
        self.G = np.array([[[1.0]]])

        x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
        self.u_exact = np.exp(-(x**2 + 2 * y**2 + 3 * z**2 + 3 * d1 * Tf))
        self.compute_error = 1

        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        zero_core = np.array([[[0.0]]])

        # Time-dependent scalar core: [t] in MATLAB → core = t * ones(1,1,1)
        self.A = lambda t: ([ones_x, (-yvals).reshape(-1, 1), ones_z], np.array([[[t]]]))
        self.B = lambda t: ([xvals.reshape(-1, 1), ones_y, ones_z], np.array([[[t]]]))
        self.C = lambda t: ([np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))], zero_core)

        Q_base = np.stack(
            [
                np.array([[9 * d1, 0, -16 * d1], [0, -2, 0], [-4 * d1, 0, 0]]),
                np.array([[-36 * d1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            ],
            axis=2,
        )

        def P1(t):
            return np.column_stack([
                np.exp(-(xvals**2)),
                xvals * np.exp(-(xvals**2)),
                (xvals**2) * np.exp(-(xvals**2)),
            ])

        def P2(t):
            return np.column_stack([
                np.exp(-2 * (yvals**2)),
                yvals * np.exp(-2 * (yvals**2)),
                (yvals**2) * np.exp(-2 * (yvals**2)),
            ])

        def P3(t):
            return np.column_stack([
                np.exp(-3 * (zvals**2)),
                (zvals**2) * np.exp(-3 * (zvals**2)),
            ])

        # Note: MATLAB test 6 has -2*t in the (1,1) slot of the first core slice
        # so the full Q = exp(-3*d1*t) * [[9d1,0,-16d1],[0,-2t,0],[-4d1,0,0]] stacked with [[-36d1,...]]
        def Q(t):
            slice0 = np.array([
                [9 * d1, 0, -16 * d1],
                [0, -2 * t, 0],
                [-4 * d1, 0, 0],
            ])
            slice1 = np.array([[-36 * d1, 0, 0], [0, 0, 0], [0, 0, 0]])
            return np.exp(-3 * d1 * t) * np.stack([slice0, slice1], axis=2)

        self.P = lambda t: ([P1(t), P2(t), P3(t)], Q(t))

    def _test7(self, Tf, Nx, Ny, Nz):
        """
        Rigid-body rotation about <0,0,1> with time-dependent flow:
          u_t - 2t*y*u_x + 2t*x*u_y = (1/12)(u_xx + u_yy + u_zz)
        Rank test (no exact solution needed).
        """
        self.L = 4 * np.pi
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([Tf * self.L, Tf * self.L, 0.0])
        d1 = d2 = d3 = 1 / 12
        self.diffcoefs = np.array([d1, d2, d3])

        self.U = [
            np.exp(-(xvals**2)).reshape(-1, 1),
            np.exp(-9 * (yvals**2)).reshape(-1, 1),
            np.exp(-(zvals**2)).reshape(-1, 1),
        ]
        self.G = np.array([[[1.0]]])
        self.u_exact = 0
        self.compute_error = 0

        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        zero_core = np.array([[[0.0]]])

        self.A = lambda t: ([ones_x, (-yvals).reshape(-1, 1), ones_z], np.array([[[2 * t]]]))
        self.B = lambda t: ([xvals.reshape(-1, 1), ones_y, ones_z], np.array([[[2 * t]]]))
        self.C = lambda t: ([np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))], zero_core)

        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core,
        )

    def _test8(self, Tf, Nx, Ny, Nz):
        """
        Dougherty–Fokker–Planck equation (isotropic relaxation toward Maxwellian):
          u_t - (xu)_x - (yu)_y - (zu)_z = (1/2)(u_xx + u_yy + u_zz)
        Rank test. Initial condition is a two-stream Maxwellian (rank 2).
        No exact solution tracked; equilibrium is exp(-x^2-y^2-z^2).
        """
        self.L = 16.0
        xvals, dx = self._cell_centers(self.L, Nx)
        yvals, dy = self._cell_centers(self.L, Ny)
        zvals, dz = self._cell_centers(self.L, Nz)
        self.xvals, self.dx = xvals, dx
        self.yvals, self.dy = yvals, dy
        self.zvals, self.dz = zvals, dz

        self.CFLconstraints = np.array([self.L / 2, self.L / 2, self.L / 2])
        d1 = d2 = d3 = 1 / 2
        self.diffcoefs = np.array([d1, d2, d3])

        # Two-stream Maxwellian parameters
        R = 1 / 6
        n1, u1x, u1y, u1z, T1 = (
            0.8037121822811545, -0.3403147128006618, 0.0, 0.0, 0.1033754314349305
        )
        n2, u2x, u2y, u2z, T2 = (
            4.764615814550553, 0.05740548475117823, 0.0, 0.0, 3.442950196134546
        )

        # Tucker IC: rank-2 Maxwellian mixture
        self.U = [
            np.column_stack([
                np.exp(-((xvals - u1x) ** 2) / (2 * R * T1)),
                np.exp(-((xvals - u2x) ** 2) / (2 * R * T2)),
            ]),
            np.column_stack([
                np.exp(-((yvals - u1y) ** 2) / (2 * R * T1)),
                np.exp(-((yvals - u2y) ** 2) / (2 * R * T2)),
            ]),
            np.column_stack([
                np.exp(-((zvals - u1z) ** 2) / (2 * R * T1)),
                np.exp(-((zvals - u2z) ** 2) / (2 * R * T2)),
            ]),
        ]
        # MATLAB: G = cat(3,[n1*(2piRT1)^(-3/2), 0; 0,0], [0,0; 0,n2*(2piRT2)^(-3/2)])
        # → shape (2,2,2): diagonal slices along axis 2
        G = np.zeros((2, 2, 2))
        G[0, 0, 0] = n1 * (2 * np.pi * R * T1) ** (-3 / 2)
        G[1, 1, 1] = n2 * (2 * np.pi * R * T2) ** (-3 / 2)
        self.G = G
        self.u_exact = 0
        self.compute_error = 0

        ones_x = np.ones((Nx, 1))
        ones_y = np.ones((Ny, 1))
        ones_z = np.ones((Nz, 1))
        core1 = np.array([[[1.0]]])
        zero_core = np.array([[[0.0]]])

        # A = -x (Fokker-Planck drift in x): factors [(-x), 1, 1], core [[1]]
        self.A = lambda t: ([(-xvals).reshape(-1, 1), ones_y, ones_z], core1)
        self.B = lambda t: ([ones_x, (-yvals).reshape(-1, 1), ones_z], core1)
        self.C = lambda t: ([ones_x, ones_y, (-zvals).reshape(-1, 1)], core1)

        self.P = lambda t: (
            [np.zeros((Nx, 1)), np.zeros((Ny, 1)), np.zeros((Nz, 1))],
            zero_core,
        )
