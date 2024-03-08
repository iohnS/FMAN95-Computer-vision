import numpy as np
import numpy.random as rand
import scipy.io
import matplotlib.image as mpimg
import cv2 as cv
from triangulate import triangulate

# https://vnav.mit.edu/material/14-2viewGeometry-notes.pdf
# https://users.cecs.anu.edu.au/~hongdong/new5pt_cameraREady_ver_1.pdf 
# https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf 

# Why do we want a five point solver when we have an eight point solver? 
# Its a minimal solver which have fewer critical surfaces

# It's much more efficient in RANSAC compared to eight point algorithm. 
# It also has better accuracy as it has checked more geometric constraints of the problem. 

# DONT WASTE YOUR TIME LIKE ME: OPENCV HAS A FUNCTION cv2.findEssentialMat() WHICH USES DAVID NISTER'S FIVE POINT ALGORITHM TO FIND ESSENTIAL MATRIX. 

def check_in_front(P1, P2s, Xs):
    largest = 0
    P2 = None
    X = None
    for i in range(0, len(P2s)):
        x1 = np.matmul(P1, Xs[i])
        x2 = np.matmul(P2s[i], Xs[i])
        inFront = 0
        for j in range(0, len(x1[2])):
            if x1[2][j] > 0 and x2[2][j] > 0:
                inFront += 1
        if inFront > largest:
            largest = inFront
            P2 = P2s[i]
            X = Xs[i]
    return P2, X

data = scipy.io.loadmat("../assignment4data/compEx2data.mat")
im1 = mpimg.imread("../assignment4data/im1.jpg")
im2 = mpimg.imread("../assignment4data/im2.jpg")
K = data['K']
[x1, x2] = data['x'][0]

hx1 = np.vstack((x1, np.ones(len(x1[0]))))
hx2 = np.vstack((x2, np.ones(len(x2[0]))))

normx1 = np.linalg.inv(K) @ hx1
normx2 = np.linalg.inv(K) @ hx2

nx1t = np.transpose(normx1)
nx2t = np.transpose(normx2)

N = 1 # The number of iterations. 
bestInliers = 0
e = 5 # The acceptable error
for i in range(0, N):
    rx1 = [nx1t[rand.randint(0, len(nx1t))][:2] for _ in range(0,5)]
    rx2 = [nx2t[rand.randint(0, len(nx2t))][:2] for _ in range(0,5)]
    E = cv.findEssentialMat(np.array(rx1), np.array(rx2), maxIters=100)[0]
    Es = [E[i:i+3] for i in range(0, len(E), 3)]
    for E in Es:
        currentInliers = 0
        ep1 = np.transpose(E @ normx1) # It looks like the lines are already normalized. 
        ep2 = np.transpose(E @ normx2)

        for i in range(0, len(ep1)):
            x1, y1, z1 = nx2t[i]
            a1, b1, c1 = ep1[i]
            dist1 = np.abs(a1 * x1 + b1 * y1 + c1) / np.sqrt(a1**2 + b1**2)
                  
            
            x2, y2, z2 = nx1t[i]
            a2, b2, c2 = ep2[i]
            dist2 = np.abs(a2 * x2 + b2 * y2 + c2) / np.sqrt(a2**2 + b2**2)
            
            if dist1 < e and dist2 < e:
                currentInliers += 1

        if currentInliers > bestInliers:
            bestInliers = currentInliers
            bestE = E

W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
WT = np.transpose(W)
Z = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
[U, S, V] = np.linalg.svd(bestE)
newE = U @ np.diag([1, 1, 0]) @ V
[U, S, V] = np.linalg.svd(newE)
u3 = [[row[2]]for row in U]
mu3 = [[-row[2]]for row in U]
P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
P2a = np.hstack([U @ W @ np.transpose(V), u3])
P2b = np.hstack([U @ W @ np.transpose(V), mu3])
P2c = np.hstack([U @ np.transpose(W) @ np.transpose(V), u3])
P2d = np.hstack([U @ np.transpose(W) @ np.transpose(V), mu3]) 

X1 = np.transpose(triangulate(P1, P2a, nx1t, nx2t))
X2 = np.transpose(triangulate(P1, P2b, nx1t, nx2t))
X3 = np.transpose(triangulate(P1, P2c, nx1t, nx2t))
X4 = np.transpose(triangulate(P1, P2d, nx1t, nx2t))

P2s = [P2a, P2b, P2c, P2d]
Xs = [X1, X2, X3, X4]
P2, X = check_in_front(P1, P2s, Xs)
