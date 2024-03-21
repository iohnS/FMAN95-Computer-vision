import scipy.io
import numpy as np
from rq import rq

data = scipy.io.loadmat("../assignment2data/compEx1data.mat")
P = data["P"][0] #  P{i} The camera matrix for the image i in imfiles{i}.

T1 = [[1,0,0,0], [0,4,0,0], [0,0,1,0], [1/10, 1/10, 0, 1]]
T2 = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [1/16, 1/16, 0, 1]]
T1inv = np.linalg.inv(T1)
T2inv = np.linalg.inv(T2)

P2T1 = np.matmul(P[1], T1inv)
P2T2 = np.matmul(P[1], T2inv)

K1 = rq(P2T1)[0]
K1 = [[e /K1[-1][-1]for e in row]for row in K1]

K2 = rq(P2T2)[0]
K2 = [[e /K2[-1, -1] for e in row]for row in K2]

print("K1:")
for row in K1:
    print(row)

print("K2:")
for row in K2:
    print(row)  