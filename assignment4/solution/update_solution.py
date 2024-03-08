import numpy as np

def update_solution(deltav, P, U):
    Ba = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    Bb = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Bc = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

    dpointvar = np.concatenate(([0], deltav[:3 * U.shape[1] - 1]))
    dpointvar = np.reshape(dpointvar, U[:3, :].shape)
    dcamvar = np.concatenate(([0, 0, 0, 0, 0, 0], deltav[3 * U.shape[1]:]))
    dcamvar = np.reshape(dcamvar, (6, len(P)))

    Unew = pextend(U[:3, :] + dpointvar)

    Pnew = []
    for i in range(len(P)):
        KR0 = P[i][:3]
        t0 = P[i][3]
        KR = np.dot(KR0, expm(Ba * dcamvar[0, i] + Bb * dcamvar[1, i] + Bc * dcamvar[2, i]))
        t = t0 + dcamvar[3:6, i]
        Pnew.append(np.column_stack((KR, t)))

    return Pnew, Unew

def pextend(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

def expm(B):
    return np.eye(3) + B + 0.5 * np.dot(B, B)
