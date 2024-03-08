from ce1 import F, N1, N2, x1, x2, kronan1, kronan2, n2t, n1t
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from pflat import pflat 
from triangulate import triangulate

def skew_symmetric(m):
    if(len(m) != 3):
        raise ValueError("Matrix must be 1x3")
    
    mx = [[0, -m[2], m[1]], [m[2], 0, -m[0]], [-m[1], m[0], 0]]
    return mx

def pflat1d(d):
    last = d[-1]
    return [e / last for e in d]

x1t = np.transpose(x1)
x2t = np.transpose(x2)


e2 = null_space(np.transpose(F))
e2x = skew_symmetric([e[0] for e in e2])
e2xF = np.matmul(e2x, F)

P2 = np.hstack([e2xF, e2])
P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
P1n = np.matmul(N1, P1)
P2n = np.matmul(N2, P2)
zeros = [[0] for _ in range(0, len(n1t[0]))]

X = triangulate(P1n, P2n, n1t, n2t)

xproj1 = np.transpose(pflat(np.linalg.inv(N1) @ (P1n @ np.transpose(X))))
xproj2 = np.transpose(pflat(np.linalg.inv(N2) @ (P2n @ np.transpose(X))))


#plt.imshow(kronan2, cmap='gray')
#plt.scatter([x[0] for x in x2t], [x[1] for x in x2t], c='r', s=2)
#plt.scatter([x[0] for x in xproj2], [x[1] for x in xproj2], c='b', s=1)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter([p[0] for p in X], [p[1] for p in X], [p[2] for p in X], c='r', s=1)

plt.show()