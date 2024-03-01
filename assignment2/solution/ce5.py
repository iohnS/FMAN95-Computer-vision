import ce3
import ce4
import matplotlib.pyplot as plt
import numpy as np
from pflat import pflat
import matplotlib.image as mpimg
import math
from sympy import Matrix
import tkinter as tk
from plotcams import plotcams

def pflat1d(d):
    last = d[-1]
    return [e / last for e in d]

hx1 = [[ce4.x1[i][0], ce4.x1[i][1], 1]for i in range(0, len(ce4.x1))]
hx2 = [[ce4.x2[i][0], ce4.x2[i][1], 1]for i in range(0, len(ce4.x2))]
zeros = [[0] for _ in range(0, len(hx1[0]))]
def DLT(x1, x2):
    X = []                  # each row is a homogenous 3d point that we have approximated with P1 and P2 together with image points x1 and x2
    for i in range(0, len(x1)):
        r1 = np.hstack((ce3.P1, [[e] for e in x1[i]], zeros)) # ITS SUPPOSED TO BE MINUS HX1
        r2 = np.hstack((ce3.P2, zeros, [[e] for e in x2[i]]))
        M = np.vstack((r1, r2))
        [U, S, V] = np.linalg.svd(M)
        X.append(pflat1d(V[-1][:4]))
    
    return X

X = DLT(hx1, hx2)

xproj1 = pflat(np.matmul(ce3.P1, np.transpose(X)))
xproj2 = pflat(np.matmul(ce3.P2, np.transpose(X)))

xproj1k = (np.linalg.inv(ce3.K1) @ xproj1)
xproj2k = (np.linalg.inv(ce3.K2) @ xproj2)
hx1k = np.transpose(np.linalg.inv(ce3.K1) @ np.transpose(hx1))
hx2k = np.transpose(np.linalg.inv(ce3.K2) @ np.transpose(hx2))

cube1 = mpimg.imread("../assignment2data/cube1.JPG")
cube2 = mpimg.imread("../assignment2data/cube2.JPG")

goodPoints = []
for i in range(0, len(xproj1[0])):
    # Should use euclidean distance but this works just as well.
    if math.dist((xproj1[0][i], xproj1[1][i]), (hx1[i][0], hx1[i][1])) < 3 and math.dist((xproj2[0][i], xproj2[1][i]), (hx2[i][0], hx2[i][1])) < 3:
        goodPoints.append(i)

P1np = np.array(Matrix(ce3.P1).nullspace()[0])      # The camera centers are negative so we have to multiply with -1
P2np = np.array(Matrix(ce3.P2).nullspace()[0])
cc1 = [e[0] for e in P1np][:3]
cc2 = [e[0] for e in P2np][:3]
pa1 = ce3.P1[2][:3]
pa2 = ce3.P2[2][:3]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal', adjustable='box')
plotcams([ce3.P1], ax)
plotcams([ce3.P2], ax)
Xg = np.transpose([X[i] for i in goodPoints])
#ax.scatter(X[0], X[1], X[2], s=0.5, c='b')
ax.scatter(Xg[0], Xg[1], Xg[2], s=0.5, c='b')
ax.scatter(ce3.Xmodel[0], ce3.Xmodel[1], ce3.Xmodel[2], s=0.5, c='r')
    
#for i in range(0, len(ep1[0])):
#    ax.plot([s[0][i], endpoints[0][i]], [startpoints[1][i], endpoints[1][i]], "b")


#plt.imshow(cube1)

#for i in range(0, len(hx2k)):
#    plt.scatter(hx2k[i][0], hx2k[i][1], c='b', s=0.1)
#    plt.scatter(xproj2k[0][i], xproj2k[1][i], c='r', s=0.1)

plt.show()