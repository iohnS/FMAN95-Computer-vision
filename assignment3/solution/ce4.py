from ce3 import E, U, V
import numpy as np 
import scipy
from pflat import pflat, pflat1d
from plotcams import plotcams  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sympy import Matrix
from triangulate import triangulate 

kronan2 = mpimg.imread("../assignment3data/kronan2.JPG")
x = scipy.io.loadmat("../assignment3data/compEx1data.mat")['x']
[x1, x2] = [x[0][0], x[1][0]] # 2008 points.
K = scipy.io.loadmat("../assignment3data/compEx3data.mat")['K']
Kinv = np.linalg.inv(K)

n1t = np.transpose(Kinv @ x1)
n2t = np.transpose(Kinv @ x2)

def doEverything(P1, P2):
    P1np = np.array(Matrix(P1).nullspace()[0])
    P2np = np.array(Matrix(P2).nullspace()[0])
    cc1 = [e[0] for e in P1np]
    cc2 = [e[0] for e in P2np]
    pa1 = P1[2][0:3]
    pa2 = P2[2][0:3]

    inFront = 0
    X = triangulate(P1, P2, n1t, n2t)
    for p in X:
        if p[2] > 0:
            inFront += 1
    fig3d = plt.figure()
    sp3d = fig3d.add_subplot(projection='3d')
    sp3d.scatter(cc1[0], cc1[1], cc1[2], c='r')
    sp3d.scatter(cc2[0], cc2[1], cc2[2], c='r')
    sp3d.quiver(cc1[0], cc1[1], cc1[2], pa1[0], pa1[1], pa1[2], color="g")
    sp3d.quiver(cc2[0], cc2[1], cc2[2], pa2[0], pa2[1], pa2[2], color="g")
    sp3d.scatter([row[0] for row in X], [row[1] for row in X], [row[2] for row in X], c='b', s=0.5)
    sp3d.set_xlabel('X')
    sp3d.set_ylabel('Y')
    sp3d.set_zlabel('Z')
    print(inFront)
    plt.show()
    

W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
WT = np.transpose(W)
u3 = [[row[2]]for row in U]
mu3 = [[-row[2]]for row in U]
P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
P2a = np.hstack([U @ W @ np.transpose(V), u3])
P2b = np.hstack([U @ W @ np.transpose(V), mu3])
P2c = np.hstack([U @ np.transpose(W) @ np.transpose(V), u3])
P2d = np.hstack([U @ np.transpose(W) @ np.transpose(V), mu3]) 

doEverything(P1, P2d)
