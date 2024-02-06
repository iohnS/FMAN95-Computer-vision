import scipy.io
import numpy as np
from pflat import pflat

data = scipy.io.loadmat("../assignment2data/compEx1data.mat")
P = data["P"][0] #  P{i} The camera matrix for the image i in imfiles{i}.

T1 = [[1,0,0,0], [0,4,0,0], [0,0,1,0], [1/10, 1/10, 0,1]]
T2 = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [1/16, 1/16, 0,1]]


P2 = P[1]

K1 = np.linalg.qr(np.matmul(P2, T1))[0]
K2 = np.linalg.qr(np.matmul(P2, T2))[0]

print("K1 matrix:")
for i in range(0, len(K1)):
    print(K1[i])

print("\n K2 matrix:")
for i in range(0, len(K2)):
    print(K2[i])