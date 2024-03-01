import scipy.io
import numpy as np
from pflat import pflat

data = scipy.io.loadmat("../assignment2data/compEx1data.mat")
P = data["P"][0] #  P{i} The camera matrix for the image i in imfiles{i}.

T1 = [[1,0,0,0], [0,4,0,0], [0,0,1,0], [1/10, 1/10, 0,1]]
T2 = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [1/16, 1/16, 0,1]]
T1inv = np.linalg.inv(T1)
T2inv = np.linalg.inv(T2)

P1 = []
for i in range(0, 9):
    P1.append(np.matmul(P[i], T1inv))
    
P2 = []
for i in range(0,9):
    P2.append(np.matmul(P[i], T2inv))

K1 = np.linalg.qr(P1[1])[0]
K1 = [[e /K1[-1, -1] for e in row]for row in K1]
K2 = np.linalg.qr(P2[1])[0]
K2 = [[e /K2[-1, -1] for e in row]for row in K2]

print("K1 matrix:")
for i in range(0, len(K1)):
    print(K1[i])

print("\n K2 matrix:")
for i in range(0, len(K2)):
    print(K2[i])