import numpy as np
from pflat import pflat1d

def triangulate(P1, P2, x1, x2):
    zeros = [[0] for _ in range(0, len(x1[0]))]
    X = []
    for i in range(0, len(x1)):
        r1 = np.hstack((P1, [[-e] for e in x1[i]], zeros))
        r2 = np.hstack((P2, zeros, [[-e] for e in x2[i]]))
        M = np.vstack((r1, r2))
        [U, S, V] = np.linalg.svd(M)
        X.append(pflat1d(V[-1][:4]))
    
    return X