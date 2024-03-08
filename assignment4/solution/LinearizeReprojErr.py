import numpy as np
from scipy.sparse import csr_matrix

def LinearizeReprojErr(P, U, u):
    numpts = U.shape[1]
    Ba = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    Bb = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Bc = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

    da1 = [None] * len(P)
    db1 = [None] * len(P)
    dc1 = [None] * len(P)
    dt11 = [None] * len(P)
    dt21 = [None] * len(P)
    dt31 = [None] * len(P)
    dU11 = [None] * len(P)
    dU21 = [None] * len(P)
    dU31 = [None] * len(P)
    da2 = [None] * len(P)
    db2 = [None] * len(P)
    dc2 = [None] * len(P)
    dt12 = [None] * len(P)
    dt22 = [None] * len(P)
    dt32 = [None] * len(P)
    dU12 = [None] * len(P)
    dU22 = [None] * len(P)
    dU32 = [None] * len(P)

    U = pextend(U)
    U = U[:3, :]

    for i in range(len(P)):
        vis = np.where(np.isfinite(u[i][0]))[0]
        KR0 = P[i][:3]
        t0 = P[i][3]
        pext_U = pextend(U[:, vis])
        
        da1[i] = (KR0[0] @ Ba @ U[:, vis]) / (KR0[2] @ U[:, vis] + t0[2]) - \
                 (KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * (KR0[2] @ Ba @ U[:, vis])

        da2[i] = (KR0[1] @ Ba @ U[:, vis]) / (KR0[2] @ U[:, vis] + t0[2]) - \
                 (KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * (KR0[2] @ Ba @ U[:, vis])

        db1[i] = (KR0[0] @ Bb @ U[:, vis]) / (KR0[2] @ U[:, vis] + t0[2]) - \
                 (KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * (KR0[2] @ Bb @ U[:, vis])

        db2[i] = (KR0[1] @ Bb @ U[:, vis]) / (KR0[2] @ U[:, vis] + t0[2]) - \
                 (KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * (KR0[2] @ Bb @ U[:, vis])

        dc1[i] = (KR0[0] @ Bc @ U[:, vis]) / (KR0[2] @ U[:, vis] + t0[2]) - \
                 (KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * (KR0[2] @ Bc @ U[:, vis])

        dc2[i] = (KR0[1] @ Bc @ U[:, vis]) / (KR0[2] @ U[:, vis] + t0[2]) - \
                 (KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * (KR0[2] @ Bc @ U[:, vis])

        dU11[i] = KR0[0, 0] / (KR0[2] @ U[:, vis] + t0[2]) - \
                  (KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * KR0[2, 0]

        dU12[i] = KR0[1, 0] / (KR0[2] @ U[:, vis] + t0[2]) - \
                  (KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * KR0[2, 0]

        dU21[i] = KR0[0, 1] / (KR0[2] @ U[:, vis] + t0[2]) - \
                  (KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * KR0[2, 1]

        dU22[i] = KR0[1, 1] / (KR0[2] @ U[:, vis] + t0[2]) - \
                  (KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * KR0[2, 1]

        dU31[i] = KR0[0, 2] / (KR0[2] @ U[:, vis] + t0[2]) - \
                  (KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * KR0[2, 2]

        dU32[i] = KR0[1, 2] / (KR0[2] @ U[:, vis] + t0[2]) - \
                  (KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2) * KR0[2, 2]

        dt11[i] = 1 / (KR0[2] @ U[:, vis] + t0[2])
        dt12[i] = np.zeros_like(dt11[i])

        dt21[i] = np.zeros_like(dt11[i])
        dt22[i] = 1 / (KR0[2] @ U[:, vis] + t0[2])

        dt31[i] = -(KR0[0] @ U[:, vis] + t0[0]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2)
        dt32[i] = -(KR0[1] @ U[:, vis] + t0[1]) / ((KR0[2] @ U[:, vis] + t0[2]) ** 2)

    row, col, data = [], [], []
    resnum = 0

    for i in range(len(P)):
        uu = u[i]
        vis = np.where(np.isfinite(uu[0]))[0]

        # For first residual:
        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend((vis * 3).reshape(-1, 1).tolist())
        data.extend(np.concatenate(dU11[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend((vis * 3 + 1).reshape(-1, 1).tolist())
        data.extend(np.concatenate(dU21[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend((vis * 3 + 2).reshape(-1, 1).tolist())
        data.extend(np.concatenate(dU31[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 1)] * len(vis))
        data.extend(np.concatenate(da1[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 2)] * len(vis))
        data.extend(np.concatenate(db1[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 3)] * len(vis))
        data.extend(np.concatenate(dc1[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 4)] * len(vis))
        data.extend(np.concatenate(dt11[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 5)] * len(vis))
        data.extend(np.concatenate(dt21[i]).tolist())

        row.extend(np.arange(resnum, resnum + 2 * len(vis), 2))
        col.extend([(3 * numpts + i * 6)] * len(vis))
        data.extend(np.concatenate(dt31[i]).tolist())

        # For second residual:
        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend((vis * 3).reshape(-1, 1).tolist())
        data.extend(np.concatenate(dU12[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend((vis * 3 + 1).reshape(-1, 1).tolist())
        data.extend(np.concatenate(dU22[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend((vis * 3 + 2).reshape(-1, 1).tolist())
        data.extend(np.concatenate(dU32[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 1)] * len(vis))
        data.extend(np.concatenate(da2[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 2)] * len(vis))
        data.extend(np.concatenate(db2[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 3)] * len(vis))
        data.extend(np.concatenate(dc2[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 4)] * len(vis))
        data.extend(np.concatenate(dt12[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend([(3 * numpts + (i - 1) * 6 + 5)] * len(vis))
        data.extend(np.concatenate(dt22[i]).tolist())

        row.extend(np.arange(resnum + 1, resnum + 2 * len(vis) + 1, 2))
        col.extend([(3 * numpts + i * 6)] * len(vis))
        data.extend(np.concatenate(dt32[i]).tolist())

        # Constants
        r1 = np.zeros(2 * len(vis))
        r1[::2] = (P[i][:, :2] @ pext_U) / (P[i][:, 2] @ pext_U) - uu[0, vis]
        r2 = np.zeros(2 * len(vis))
        r2[1::2] = (P[i][:, :2] @ pext_U) / (P[i][:, 2] @ pext_U) - uu[1, vis]

        r = np.concatenate([r1, r2])

        resnum += 2 * len(vis)

    A = csr_matrix((data, (row, col)), shape=(2 * resnum, 3 * numpts + 6 * len(P))) # This is the jacobian matrix that we construct.

    return r, A

def pextend(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))
