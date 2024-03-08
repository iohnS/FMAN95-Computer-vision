import numpy as np

def ComputeReprojectionError(P, U, u):
    err = 0
    res = []
    for i in range(len(P)):
        uu = u[i]
        vis = np.isfinite(uu[0])
        err += np.sum(((P[i][0] @ U[:, vis]) / (P[i][2] @ U[:, vis]) - uu[0, vis])**2) + \
               np.sum(((P[i][1] @ U[:, vis]) / (P[i][2] @ U[:, vis]) - uu[1, vis])**2)
        res.append(((P[i][0] @ U[:, vis]) / (P[i][2] @ U[:, vis]) - uu[0, vis])**2 + \
                   ((P[i][1] @ U[:, vis]) / (P[i][2] @ U[:, vis]) - uu[1, vis])**2)
    return err, np.concatenate(res)
