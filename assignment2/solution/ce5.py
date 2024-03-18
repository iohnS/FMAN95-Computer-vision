import ce3
import ce4
import matplotlib.pyplot as plt
import numpy as np
from pflat import pflat
import matplotlib.image as mpimg
import math
from sympy import Matrix
from plotcams import plotcams
from triangulate import triangulate

cube1 = mpimg.imread("../assignment2data/cube1.JPG")
cube2 = mpimg.imread("../assignment2data/cube2.JPG")
x1 = ce4.x1
x2 = ce4.x2
K1 = ce3.K1
K2 = ce3.K2
P1 = ce3.P1
P2 = ce3.P2
hx1 = [[x1[i][0], x1[i][1], 1]for i in range(0, len(x1))]
hx2 = [[x2[i][0], x2[i][1], 1]for i in range(0, len(x2))]
zeros = [[0] for _ in range(0, len(hx1[0]))]


X = triangulate(P1, P2, hx1, hx2)

xproj1 = pflat(P1 @ np.transpose(X))
xproj2 = pflat(P2 @ np.transpose(X))

# Computed points and SIFT points:
#plt.imshow(cube2)
#plt.scatter(xproj2[0], xproj2[1], s=1, c='r')
#plt.scatter(np.transpose(hx2)[0], np.transpose(hx2)[1], s=1, c='b')

xpk1 = pflat(np.linalg.inv(K1) @ np.transpose(hx1))
xpk2 = pflat(np.linalg.inv(K2) @ np.transpose(hx2))
P1K = np.linalg.inv(K1) @ P1
P2K = np.linalg.inv(K2) @ P2

Xk = triangulate(P1K, P2K, np.transpose(xpk1), np.transpose(xpk2))

xproj1k = pflat(P1 @ np.transpose(Xk))
xproj1k = np.transpose(list(filter(lambda x: (x[0] > 0 and x[0] < 1930), np.transpose(xproj1k))))
xproj2k = pflat(P2 @ np.transpose(Xk))
xproj2k = np.transpose(list(filter(lambda x: (x[0] > 0 and x[0] < 1930), np.transpose(xproj2k))))

# Computed points and SIFT points with normalization:
#plt.imshow(cube2)
#plt.scatter(np.transpose(hx2)[0], np.transpose(hx2)[1], s=1, c='b')
#plt.scatter(xproj2k[0], xproj2k[1], s=1, c='r')

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
plotcams([P1], ax)
plotcams([P2], ax)
Xg = np.transpose([Xk[i] for i in goodPoints])
ax.scatter(Xg[0], Xg[1], Xg[2], s=0.5, c='b')
ax.scatter(ce3.Xmodel[0], ce3.Xmodel[1], ce3.Xmodel[2], s=0.5, c='r')
    
#for i in range(0, len(ep1[0])):
#    ax.plot([s[0][i], endpoints[0][i]], [startpoints[1][i], endpoints[1][i]], "b")

plt.show()