import numpy as np

def fivepoint_solver(x1n, x2n):
    # Computes Essential matrices from the normalized image coordinates x1n and x2n
    # x1n should be 3x5 where each column corresponds to a homogeneous point
    # x2n should have the same format.
    M = np.zeros((5, 9))
    for i in range(5):
        xx = np.outer(x1n[:, i], x2n[:, i])
        M[i, :] = xx.ravel()

    Evec = np.linalg.svd(M.T @ M)[2]
    E = [Evec[:, i].reshape(3, 3).T for i in range(4)]

    # The constraint: 2*E*E'*E - trace(E*E')*E = 0
    coeffs = np.zeros((9, 64))
    mons = np.zeros((4, 64))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                mons[i, k + (j - 1) * 4 + (i - 1) * 16] += 1
                mons[j, k + (j - 1) * 4 + (i - 1) * 16] += 1
                mons[k, k + (j - 1) * 4 + (i - 1) * 16] += 1

                new_coeffs = 2 * E[i] @ E[j].T @ E[k] - np.trace(E[i] @ E[j].T) * E[k]
                coeffs[:, k + (j - 1) * 4 + (i - 1) * 16] = new_coeffs.ravel()

    # The determinant constraint
    det_coeffs = np.zeros(64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                det_coeffs[k + (j - 1) * 4 + (i - 1) * 16] = (
                    E[i][0, 0] * E[j][1, 1] * E[k][2, 2]
                    + E[i][0, 1] * E[j][1, 2] * E[k][2, 0]
                    + E[i][0, 2] * E[j][1, 0] * E[k][2, 1]
                    - E[i][0, 0] * E[j][1, 2] * E[k][2, 1]
                    - E[i][0, 1] * E[j][1, 0] * E[k][2, 2]
                    - E[i][0, 2] * E[j][1, 1] * E[k][2, 0]
                )

    coeffs = np.vstack((coeffs, det_coeffs))

    # Make monomials unique
    _, I, J = np.unique(mons.T, axis=0, return_index=True)
    mons = mons[:, I]
    coeffs_small = np.zeros((10, len(I)))
    for i in range(coeffs.shape[0]):
        coeffs_small[i, :] = np.bincount(J, weights=coeffs[i, :])

    # Set x_1 = 1
    mons = mons[1:4, :]

    # Reduced row echelon form
    reduced_coeffs = np.linalg.qr(coeffs_small.T, mode='r')[1]

    # Construct action matrix multiplication with x_4
    Mx4 = np.zeros((10, 10))
    mon_basis = mons
    for i in range(mon_basis.shape[1]):
        x4mon = mon_basis[:, i] + np.array([0, 0, 1])
        if np.sum(x4mon) >= 3:
            row = np.where(np.sum(np.abs(mons[:, :10] - x4mon[:, np.newaxis]), axis=0) == 0)[0][0]
            Mx4[:, i] = -reduced_coeffs[row, 10:]
        else:
            ind = np.where(np.sum(np.abs(mon_basis - x4mon[:, np.newaxis]), axis=0) == 0)[0][0]
            Mx4[ind, i] = 1

    # Extract real solutions
    V, D = np.linalg.eig(Mx4.T)
    V = V / V[-1]

    Esol = []
    for i in range(10):
        if np.isreal(D[i, i]):
            x2 = V[-2, i]
            x3 = V[-3, i]
            x4 = V[-4, i]
            Esol.append(E[0] + E[1] * x2 + E[2] * x3 + E[3] * x4)

    return Esol