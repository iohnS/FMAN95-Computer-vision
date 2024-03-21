from ce3 import E
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pflat import pflat
import functions as f
from triangulate import triangulate

kronan1 = mpimg.imread("../assignment3data/kronan1.JPG")
kronan2 = mpimg.imread("../assignment3data/kronan2.JPG")
x = scipy.io.loadmat("../assignment3data/compEx1data.mat")['x']
[x1, x2] = [x[0][0], x[1][0]] # 2008 points.
K = scipy.io.loadmat("../assignment3data/compEx3data.mat")['K']
Kinv = np.linalg.inv(K)
n1t = np.transpose(Kinv @ x1)
n2t = np.transpose(Kinv @ x2)

[U, S, V] = np.linalg.svd(E)
    
P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]  
W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
WT = np.transpose(W)
u3 = [[row[2]]for row in U]
mu3 = [[-row[2]]for row in U]
P2a = np.hstack([U @ W @ V, u3])
P2b = np.hstack([U @ W @ V, mu3])
P2c = np.hstack([U @ np.transpose(W) @ V, u3])
P2d = np.hstack([U @ np.transpose(W) @ V, mu3]) 

Ps = [P2a, P2b, P2c, P2d]

#[P2_best, X_best] = f.getBest(Ps, n1t, n2t)

#f.show3d(P1, P2a, n1t, n2t)
#f.show3d(P1, P2b, n1t, n2t)
#f.show3d(P1, P2c, n1t, n2t)
#f.show3d(P1, P2d, n1t, n2t)

X = triangulate(P1, P2c, n1t, n2t)

P2 = K @ P2c
P1 = K @ P1
xproj1 = pflat(P1 @ np.transpose(X))
xproj2 = pflat(P2 @ np.transpose(X))

plt.imshow(kronan2)
plt.scatter(x2[0], x2[1], c='b', s=1)
plt.scatter(xproj2[0], xproj2[1], c='r', s=0.5)

plt.show()