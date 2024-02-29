import numpy as np
from scipy.linalg import null_space
from itertools import product

def fivepoint_solver(x1n, x2n):
    M = np.zeros((5, 9))
    for i in range(5):
        xx = np.outer(x1n[:, i], x2n[:, i])
        M[i, :] = xx.flatten()

    Evec = null_space(M)
    E = [np.reshape(Evec[:, i], (3, 3)).T for i in range(4)]

    coeffs = np.zeros((9, 64))
    mons = np.zeros((4, 64))
    for i, j, k in product(range(4), repeat=3):
        mons[i, k + (j - 1) * 4 + (i - 1) * 16] += 1
        mons[j, k + (j - 1) * 4 + (i - 1) * 16] += 1
        mons[k, k + (j - 1) * 4 + (i - 1) * 16] += 1

        new_coeffs = 2 * np.dot(np.dot(E[i], E[j].T), E[k]) - np.trace(np.dot(E[i], E[j].T)) * E[k]
        coeffs[:, k + (j - 1) * 4 + (i - 1) * 16] = new_coeffs.flatten()

    det_coeffs = np.zeros(64)
    for i, j, k in product(range(4), repeat=3):
        det_coeffs[k + (j - 1) * 4 + (i - 1) * 16] = (
            E[i][0, 0] * E[j][1, 1] * E[k][2, 2] + 
            E[i][0, 1] * E[j][1, 2] * E[k][2, 0] + 
            E[i][0, 2] * E[j][1, 0] * E[k][2, 1] - 
            E[i][0, 0] * E[j][1, 2] * E[k][2, 1] - 
            E[i][0, 1] * E[j][1, 0] * E[k][2, 2] - 
            E[i][0, 2] * E[j][1, 1] * E[k][2, 0]
        )

    coeffs = np.vstack((coeffs, det_coeffs))
    _, I, J = np.unique(mons.T, axis=0, return_index=True, return_inverse=True)
    mons = mons[:, I]
    coeffs_small = np.array([np.bincount(J, weights=coeffs[i, :]) for i in range(10)])

    mons = mons[1:, :]
    reduced_coeffs = np.linalg.matrix_rank(coeffs_small)

    Mx4 = np.zeros((10, 10))
    mon_basis = mons[:, 10:]
    for i in range(mon_basis.shape[1]):
        x4mon = mon_basis[:, i] + np.array([0, 0, 1])
        if np.sum(x4mon) >= 3:
            row = np.where(np.sum(np.abs(mons[:, :10] - x4mon[:, np.newaxis]), axis=0) == 0)[0]
            Mx4[:, i] = -reduced_coeffs[row, 10:]
        else:
            ind = np.where(np.sum(np.abs(mon_basis - x4mon[:, np.newaxis]), axis=0) == 0)[0]
            Mx4[ind, i] = 1

    V, D = np.linalg.eig(Mx4.T)
    V /= V[-1]

    Esol = []
    for i in range(10):
        if np.isreal(D[i, i]):
            x2 = V[-2, i]
            x3 = V[-3, i]
            x4 = V[-4, i]
            Esol.append(E[0] + E[1] * x2 + E[2] * x3 + E[3] * x4)

    return Esol
