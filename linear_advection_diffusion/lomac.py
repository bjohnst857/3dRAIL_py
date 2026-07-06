"""LoMaC conservative truncation and macroscopic (physical) quantities.

This ports the conservation machinery of the MATLAB 3dRAIL code:

    MATLAB file / section          ->  here
    ----------------------------------------------------------
    lomac_0.m / lomac_01.m / lomac_012.m  ->  lomac  (one level-parameterized fn)
    main.m  "Macroscopic quantities" ->  macroscopic_quantities
    main.m  "Weight function ..."   ->  build_lomac_data
    (moment evaluations in main.m)  ->  integral_moment

The three MATLAB LoMaC files are the same algorithm with a different number of
conserved moments, selected here by ``level``:

    level 0  (lomac_0.m)    conserve mass                      basis w*[1]
    level 1  (lomac_01.m)   conserve mass, momentum            basis w*[1, s]
    level 2  (lomac_012.m)  conserve mass, momentum, energy    basis w*[1, s, s^2]

Background
----------
The plain truncation (``helpers.nonconstrun``) keeps the solution low rank but
does *not* preserve its physical invariants.  For a transport/diffusion problem
the relevant invariants are the first three velocity moments of the solution
``u``:

    mass / number density   rho = integral of  u            dV
    momentum (bulk vel.)    J_x = integral of  x * u        dV   (and y, z)
    energy                  k   = integral of (x^2+y^2+z^2)/2 * u dV

LoMaC ("Local Macroscopic Conservation", Guo & Qiu, JSC 2024) splits the
solution as ``u = f1 + f2`` where ``f1`` lives in a fixed low-dimensional
"macroscopic" weight basis ``w(x) * [1, x, x^2]`` carrying exactly those
moments, truncates only the remainder ``f2``, and then re-attaches a macroscopic
part built from the *conserved reference* moments (those of the initial
condition).  The net effect is a low-rank solution whose mass/momentum/energy
equal the reference values to machine precision -- at the cost of a few extra
basis vectors (the multilinear rank is larger than with ``nonconstrun``).

A note on moment evaluation (avoids the C-vs-F unfolding pitfall)
----------------------------------------------------------------
A separable moment of a Tucker tensor,
``integral of a(x) b(y) c(z) * u dV``, equals

    dV * sum_{p,q,r} core[p,q,r] (a . U1)_p (b . U2)_q (c . U3)_r

where ``a . U1`` is the length-r1 vector of column sums of ``U1`` weighted by
``a`` at the grid points.  We evaluate this with ``numpy.einsum`` directly (see
:func:`integral_moment`), which is unambiguous about index order -- unlike the
MATLAB ``tens2mat(G,1) * kron(...)`` form, whose column ordering does not carry
over to NumPy's C-ordered ``tens2mat``.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pytensorlab as tl

from helpers import nonconstrun, tensoraddition, tucker_full


def integral_moment(factors, core, wx, wy, wz, dV):
    """Separable moment ``integral of wx(x) wy(y) wz(z) * u dV`` of a Tucker tensor.

    Parameters
    ----------
    factors : list[ndarray]   [U1, U2, U3], shapes (N1,r1), (N2,r2), (N3,r3)
    core    : ndarray         shape (r1, r2, r3)
    wx, wy, wz : ndarray      per-mode grid weights, lengths N1, N2, N3
    dV      : float           cell volume dx*dy*dz

    Returns
    -------
    float
    """
    a = wx @ factors[0]      # (r1,)
    b = wy @ factors[1]      # (r2,)
    c = wz @ factors[2]      # (r3,)
    return dV * np.einsum("pqr,p,q,r->", core, a, b, c)


def macroscopic_quantities(factors, core, data):
    """Mass, momentum (x,y,z) and energy of a Tucker tensor ``u``.

    Port of the macroscopic-quantity evaluations in ``main.m`` (number density,
    bulk velocity, energy).

    Parameters
    ----------
    factors, core : Tucker tensor for u
    data : LomacData    holds the grids and cell volume

    Returns
    -------
    (rho, Jx, Jy, Jz, energy) : tuple of float
    """
    x, y, z = data.xvals, data.yvals, data.zvals
    ox, oy, oz = data.ones_x, data.ones_y, data.ones_z
    dV = data.dV

    rho = integral_moment(factors, core, ox, oy, oz, dV)
    Jx = integral_moment(factors, core, x, oy, oz, dV)
    Jy = integral_moment(factors, core, ox, y, oz, dV)
    Jz = integral_moment(factors, core, ox, oy, z, dV)
    energy = (integral_moment(factors, core, x**2 / 2, oy, oz, dV)
              + integral_moment(factors, core, ox, y**2 / 2, oz, dV)
              + integral_moment(factors, core, ox, oy, z**2 / 2, dV))
    return rho, Jx, Jy, Jz, energy


def equilibrium_maxwellian(xvals, yvals, zvals, dx, dy, dz, rhoM, JxM, JyM, JzM, kM):
    """Discrete equilibrium Maxwellian matching the moments (rhoM, JxM, JyM, JzM, kM).

    Quadrature-corrected moment matching: Newton solve for (n, ux, uy, uz, T) in

        f_inf(x,y,z) = n/(2*pi*R*T)^1.5 * exp(-((x-ux)^2+(y-uy)^2+(z-uz)^2)/(2*R*T))

    such that the discrete moments of f_inf, evaluated with the same
    cell-center quadrature rule used elsewhere, equal the given targets.  Used
    by test 8 (Dougherty-Fokker-Planck) to track ``||f - f_inf||_1`` decay to
    equilibrium.  Port of ``QCM.m`` [Taitano, Chacon, Simakov, JSC 2024]; the
    Jacobian there mixes an absolute-coordinate energy residual with a
    mean-centered energy weight in its own row -- reproduced verbatim rather
    than "fixed", since it is only a Newton direction and the loop iterates to
    the residual tolerance regardless.

    Returns
    -------
    finf : ndarray, shape (Nx, Ny, Nz)
    """
    x, y, z = np.meshgrid(xvals, yvals, zvals, indexing="ij")
    dV = dx * dy * dz
    tol2, tol3 = 1.0e-13, 1.0e-15
    R = 1.0 / 6.0
    Mk = np.array([np.pi ** 1.5, 0.0, 0.0, 0.0, 3.0])

    def residual(Mk):
        n_k, ux_k, uy_k, uz_k, T_k = Mk
        r2 = (x - ux_k) ** 2 + (y - uy_k) ** 2 + (z - uz_k) ** 2
        f = (n_k / (2 * np.pi * R * T_k) ** 1.5) * np.exp(-r2 / (2 * R * T_k))
        return np.array([
            rhoM - dV * np.sum(f),
            JxM - dV * np.sum(f * x),
            JyM - dV * np.sum(f * y),
            JzM - dV * np.sum(f * z),
            kM - dV * np.sum(f * (x ** 2 + y ** 2 + z ** 2) / 2),
        ])

    Rk = residual(Mk)
    R0_norm = np.linalg.norm(Rk)
    count = 0
    while np.linalg.norm(Rk) > tol2 * R0_norm + tol3:
        n_k, ux_k, uy_k, uz_k, T_k = Mk
        r2 = (x - ux_k) ** 2 + (y - uy_k) ** 2 + (z - uz_k) ** 2
        expo = np.exp(-r2 / (2 * R * T_k))
        f = n_k / (2 * np.pi * R * T_k) ** 1.5 * expo

        dfdn = f / n_k
        dfdux = f * (x - ux_k) / (R * T_k)
        dfduy = f * (y - uy_k) / (R * T_k)
        dfduz = f * (z - uz_k) / (R * T_k)
        dfdT = n_k / (2 * np.pi * R) ** 1.5 * expo * (
            -3 / (2 * T_k ** 2.5) + r2 / (2 * R * T_k ** 3.5)
        )

        weights = [np.ones_like(x), x, y, z, r2 / 2]
        derivs = [dfdn, dfdux, dfduy, dfduz, dfdT]
        Jk = -dV * np.array([[np.sum(w * d) for d in derivs] for w in weights])

        Mk = Mk - np.linalg.solve(Jk, Rk)
        Rk = residual(Mk)
        count += 1
        if count > 10:
            break

    n_k, ux_k, uy_k, uz_k, T_k = Mk
    r2 = (x - ux_k) ** 2 + (y - uy_k) ** 2 + (z - uz_k) ** 2
    return (n_k / (2 * np.pi * R * T_k) ** 1.5) * np.exp(-r2 / (2 * R * T_k))


@dataclass
class LomacData:
    """Fixed data needed for LoMaC truncation and moment tracking (one run).

    Built once per run by :func:`build_lomac_data`.  Bundles the grid, the
    weight-function moment basis and its precomputed moment sums, the two scalar
    constants ``c`` and ``cc`` from ``main.m``, and the conserved reference
    moments (those of the initial condition).
    """
    # number of conserved moments: 0=mass, 1=+momentum, 2=+energy
    level: int
    # grid
    xvals: np.ndarray
    yvals: np.ndarray
    zvals: np.ndarray
    ones_x: np.ndarray
    ones_y: np.ndarray
    ones_z: np.ndarray
    dV: float
    # weight-function macroscopic basis  w(s) * [1, s, s^2]  per mode, shape (N,3)
    Vx_f1: np.ndarray
    Vy_f1: np.ndarray
    Vz_f1: np.ndarray
    # denominators that normalize each moment into the weight basis (constants)
    D_rho: float
    D_Jx: float
    D_Jy: float
    D_Jz: float
    cc: float
    c: float
    # conserved reference moments (from the initial condition)
    rhoM: float
    JxM: float
    JyM: float
    JzM: float
    kM: float


def build_lomac_data(p, U0, G0, level=2):
    """Assemble :class:`LomacData` from a :class:`TestParameters` and the IC.

    Ports the "Weight function for LoMaC" block of ``main.m`` (the weight ``w``,
    the constants ``c`` and ``cc``) and seeds the conserved reference moments
    ``rhoM, JxM, JyM, JzM, kM`` from the initial condition ``(U0, G0)``.

    ``level`` (0, 1, or 2) selects how many moments LoMaC will conserve; the
    full 3-column weight basis and all constants are always built (cheap), so
    the same bundle can also be used purely to *track* the quantities.
    """
    x, y, z = p.xvals, p.yvals, p.zvals
    Nx, Ny, Nz = p.Nx, p.Ny, p.Nz
    dV = p.dx * p.dy * p.dz
    ox, oy, oz = np.ones(Nx), np.ones(Ny), np.ones(Nz)

    # Weight function w = w1*w2*w3, sufficiently decaying at the boundary (s=1).
    s = 1.0
    w1, w2, w3 = np.exp(-s * x**2), np.exp(-s * y**2), np.exp(-s * z**2)
    Sw1, Sw2, Sw3 = w1.sum(), w2.sum(), w3.sum()
    Sx2w1, Sy2w2, Sz2w3 = (x**2 * w1).sum(), (y**2 * w2).sum(), (z**2 * w3).sum()

    # Macroscopic basis  w(s) * [1, s, s^2]  for each mode, shape (N, 3).
    Vx_f1 = w1[:, None] * np.column_stack([ox, x, x**2])
    Vy_f1 = w2[:, None] * np.column_stack([oy, y, y**2])
    Vz_f1 = w3[:, None] * np.column_stack([oz, z, z**2])

    # Scalar constant c  (main.m).
    c = (Sx2w1 * Sw2 * Sw3 + Sw1 * Sy2w2 * Sw3 + Sw1 * Sw2 * Sz2w3) / (Sw1 * Sw2 * Sw3)

    # Scalar constant cc: a fixed 3x3x3 core contracted with the [1, s^2, s^4]
    # weight moments of each mode (main.m).
    core_cc = np.zeros((3, 3, 3))
    core_cc[:, :, 0] = [[c**2, -2 * c, 1], [-2 * c, 2, 0], [1, 0, 0]]
    core_cc[:, :, 1] = [[-2 * c, 2, 0], [2, 0, 0], [0, 0, 0]]
    core_cc[:, :, 2] = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    Wx = np.array([Sw1, Sx2w1, (x**4 * w1).sum()])
    Wy = np.array([Sw2, Sy2w2, (y**4 * w2).sum()])
    Wz = np.array([Sw3, Sz2w3, (z**4 * w3).sum()])
    cc = dV * np.einsum("ijk,i,j,k->", core_cc, Wx, Wy, Wz)

    # Conserved reference moments = moments of the initial condition.
    ones_data = dict(xvals=x, yvals=y, zvals=z, ones_x=ox, ones_y=oy, ones_z=oz, dV=dV)
    rhoM, JxM, JyM, JzM, kM = _moments_with(U0, G0, ones_data)

    return LomacData(
        level=level,
        xvals=x, yvals=y, zvals=z, ones_x=ox, ones_y=oy, ones_z=oz, dV=dV,
        Vx_f1=Vx_f1, Vy_f1=Vy_f1, Vz_f1=Vz_f1,
        D_rho=dV * Sw1 * Sw2 * Sw3,
        D_Jx=dV * Sx2w1 * Sw2 * Sw3,
        D_Jy=dV * Sw1 * Sy2w2 * Sw3,
        D_Jz=dV * Sw1 * Sw2 * Sz2w3,
        cc=cc, c=c,
        rhoM=rhoM, JxM=JxM, JyM=JyM, JzM=JzM, kM=kM,
    )


def _moments_with(factors, core, d):
    """Helper: macroscopic moments using a plain dict of grid data (for setup)."""
    x, y, z, ox, oy, oz, dV = (d["xvals"], d["yvals"], d["zvals"],
                               d["ones_x"], d["ones_y"], d["ones_z"], d["dV"])
    rho = integral_moment(factors, core, ox, oy, oz, dV)
    Jx = integral_moment(factors, core, x, oy, oz, dV)
    Jy = integral_moment(factors, core, ox, y, oz, dV)
    Jz = integral_moment(factors, core, ox, oy, z, dV)
    k = (integral_moment(factors, core, x**2 / 2, oy, oz, dV)
         + integral_moment(factors, core, ox, y**2 / 2, oz, dV)
         + integral_moment(factors, core, ox, oy, z**2 / 2, dV))
    return rho, Jx, Jy, Jz, k


def _macro_core(rho, Jx, Jy, Jz, k, data):
    """Build the macroscopic core S that places given moments in the weight basis.

    Shared by ``S_f1``, ``S_f2P`` and ``S_fM`` in the MATLAB LoMaC files.  Its
    size and contents depend on ``data.level``: a (L+1)^3 core encoding mass
    (level 0), plus momentum (level 1), plus energy (level 2) in the basis
    ``w(s) * [1, s, s^2][:L+1]``.
    """
    L = data.level
    n = L + 1
    S = np.zeros((n, n, n))
    S[0, 0, 0] = rho / data.D_rho                 # mass (always)
    if L >= 1:                                     # momentum
        S[1, 0, 0] = Jx / data.D_Jx
        S[0, 1, 0] = Jy / data.D_Jy
        S[0, 0, 1] = Jz / data.D_Jz
    if L >= 2:                                     # energy
        energy_coeff = (2 * k - data.c * rho) / data.cc
        S[0, 0, 0] -= energy_coeff * data.c
        S[2, 0, 0] = energy_coeff
        S[0, 2, 0] = energy_coeff
        S[0, 0, 2] = energy_coeff
    return S


def lomac(factors, core, tol, data):
    """LoMaC truncation conserving the first ``data.level``+1 moments.

    Port of ``lomac_0.m`` / ``lomac_01.m`` / ``lomac_012.m`` (selected by
    ``data.level`` = 0, 1, or 2).  Truncates the Tucker tensor ``(factors, core)``
    to relative tolerance ``tol`` while forcing its mass (and, for higher levels,
    momentum and energy) to equal the conserved reference moments in ``data``.

    Returns
    -------
    new_factors : list[ndarray]
    new_core    : ndarray
    rank        : list[int]   the new multilinear rank
    """
    L = data.level
    f1_basis = [data.Vx_f1[:, :L + 1], data.Vy_f1[:, :L + 1], data.Vz_f1[:, :L + 1]]

    # Macroscopic part f1 of the current solution (carries its moments).
    rhoH, JxH, JyH, JzH, kH = macroscopic_quantities(factors, core, data)
    S_f1 = _macro_core(rhoH, JxH, JyH, JzH, kH, data)

    # f2 = f - f1, then truncate f2 to tol (plain HOSVD, like nonconstrun).
    f2_factors, f2_core = tensoraddition(factors, core, f1_basis, -S_f1)
    SU_f2, SG_f2, _ = nonconstrun(f2_factors, f2_core, tol)

    # Macroscopic part of the truncated f2 (to be subtracted back out).
    rho2, Jx2, Jy2, Jz2, k2 = macroscopic_quantities(SU_f2, SG_f2, data)
    S_f2P = _macro_core(rho2, Jx2, Jy2, Jz2, k2, data)

    # Conserved macroscopic part fM (from the reference moments).
    S_fM = _macro_core(data.rhoM, data.JxM, data.JyM, data.JzM, data.kM, data)

    # Final solution: trunc(f2) + (fM - macroscopic-part-of-trunc(f2)).  This
    # replaces f2's macroscopic content with the conserved reference.
    U_sum, G_sum = tensoraddition(f1_basis, S_fM - S_f2P, SU_f2, SG_f2)

    # Re-orthonormalize the stacked bases and put the core in canonical form.
    Vx, Rx = np.linalg.qr(U_sum[0])
    Vy, Ry = np.linalg.qr(U_sum[1])
    Vz, Rz = np.linalg.qr(U_sum[2])
    small = tucker_full([Rx, Ry, Rz], G_sum)       # lmlragen({Rx,Ry,Rz}, G)
    # The stacked f1/f2 bases overlap, so ``small`` is usually rank-deficient;
    # the full MLSVD then warns that its column space is smaller than requested.
    # That is expected here (the surplus directions get ~zero singular values),
    # so we silence just that one message.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Matrix has lower rank")
        tucker, _ = tl.mlsvd(small)                 # full MLSVD (no truncation)
    SU, SG = list(tucker.factors), tucker.core
    new_factors = [Vx @ SU[0], Vy @ SU[1], Vz @ SU[2]]
    return new_factors, SG, list(SG.shape)


def truncate(factors, core, tol, lomac_data=None):
    """Truncation dispatcher used by every IMEX step.

    With ``lomac_data=None`` this is the plain, non-conservative HOSVD truncation
    (:func:`helpers.nonconstrun`); with a :class:`LomacData` bundle it is the
    conservative :func:`lomac` at the bundle's ``level``.  This mirrors choosing
    the truncation line in the MATLAB ``IMEX*.m`` files.
    """
    if lomac_data is None:
        return nonconstrun(factors, core, tol)
    return lomac(factors, core, tol, lomac_data)
