import numpy as np
from scipy.linalg import schur, solve_sylvester
import pytensorlab as tl


def simoncini(A1, A2, A3, M1, M2, H1, H3, B, check_res=False):
    """
    Python translation of Simoncini's direct solver for third-order tensor
    linear equations.

    Reference:
        V. Simoncini, "Numerical solution of a class of third order tensor
        linear equations", Bollettino dell'Unione Matematica Italiana (BUMI),
        2020. Theorem 2.1, Algorithm T3-sylv.

    Solves:
        G * x_final - f = 0
    where:
        G = kron(kron(M1,A1),H1) + kron(kron(A2,M2),H1) + kron(kron(H3,M2),A3)
        f = vec(B)

    Parameters
    ----------
    A1, A2, A3 : ndarray (n x n)
    M1, M2     : ndarray (n x n)
    H1, H3     : ndarray (n x n)
    B          : ndarray (n x n x n)  right-hand side tensor
    check_res  : bool, optional

    Returns
    -------
    x_final : ndarray (n^3,)   vectorized solution
    X       : ndarray (n x n x n)  solution as tensor
    """

    # Schur decomposition of  A3' / H1'
    # MATLAB: [Q, R] = schur(A3' / H1')
    # => schur( solve(H1, A3).T )
    # scipy schur returns (T, Z) where T is upper triangular (MATLAB R)
    # and Z is the unitary matrix (MATLAB Q)
    M_schur = np.linalg.solve(H1, A3).T
    R, Q = schur(M_schur, output='real')   # R=upper triangular, Q=unitary

    # Unfold B along mode 0 (MATLAB mode 1). Must use Fortran (column-major)
    # order to match MATLAB's tens2mat(B,1) and the order='F' reshapes below;
    # tl.tens2mat uses C-order, which would solve a different tensor equation.
    bb = np.asarray(B).reshape(B.shape[0], -1, order='F')    # shape: n x n^2

    n1, n2 = A1.shape

    # g = (bb' / H1') * Q  = bb.T @ inv(H1.T) @ Q = solve(H1, bb).T @ Q
    g = np.linalg.solve(H1, bb).T @ Q
    ng = g.shape[1]

    zz = np.zeros((n1 * n2, ng))
    M2_inv_A1 = np.linalg.solve(M2, A1)    # M2 \ A1, reused every iteration
    M1_inv_T  = np.linalg.inv(M1.T)        # inv(M1'), reused every iteration

    for k in range(ng):
        # g_k = vec(G_k), reshape into (n2 x n2)
        G_k = g[:, k].reshape(n2, n2, order='F')

        # F = M2 \ G_k / M1'
        F = np.linalg.solve(M2, G_k) @ M1_inv_T

        if k > 0:
            # W = reshape(zz[:,0:k] * R[0:k, k], n1, n2)
            W = (zz[:, :k] @ R[:k, k]).reshape(n1, n2, order='F')
            F -= (W @ H3.T) @ M1_inv_T

        # MATLAB: lyap(M2\A1, (R(k,k)*H3' + A2')/M1', -F)
        # lyap(P, Q, C) solves P*X + X*Q = -C
        # => solve_sylvester(P, Q, -C) with C = -F => RHS = F
        P     = M2_inv_A1
        Q_syl = (R[k, k] * H3.T + A2.T) @ M1_inv_T
        Zhat  = solve_sylvester(P, Q_syl, F)

        zz[:, k] = Zhat.reshape(n1 * n2, order='F')

    # xx = Q * zz'   shape: (ng x n1*n2)
    xx = Q @ zz.T

    x_final = np.real(xx).flatten(order='F')
    X = np.real(xx).reshape(n1, n2, ng, order='F')

    if check_res:
        bb_vec   = bb.flatten(order='F')
        norm_rhs = np.linalg.norm(bb_vec)
        L_op = (np.kron(np.kron(M1, A1), H1)
              + np.kron(np.kron(A2, M2), H1)
              + np.kron(np.kron(H3, M2), A3))
        norm_res = np.linalg.norm(L_op @ x_final - bb_vec) / norm_rhs
        print(f"Final full relative residual: {norm_res:.6e}")

    return x_final, X