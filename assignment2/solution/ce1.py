import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pflat import pflat
from plotcams import plotcams

data = scipy.io.loadmat("../assignment2data/compEx1data.mat")
X = data["X"] # Homogeunous coordinates for all the 3D points. Dimension of 4 x 9471. The observed points in real world 3D space. 
x = data["x"][0] # x{i} contains the homogenous coordinates of the image points seen in image i. Dimensions 3 x 9471. 
P = data["P"][0] #  P{i} The camera matrix for the image i in imfiles{i}.
imfiles = data["imfiles"][0] #  imfiles{i}The name of the image i. 

T1 = [[1,0,0,0], [0,4,0,0], [0,0,1,0], [1/10, 1/10, 0,1]]
T2 = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [1/16, 1/16, 0,1]]
T1inv = np.linalg.inv(T1)
T2inv = np.linalg.inv(T2)

P1 = []
for i in range(0, 9):
    P1.append(np.matmul(P[i], T1))
    
P2 = []
for i in range(0,9):
    P2.append(np.matmul(P[i], T2))


def plotReconstruction(X, P):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[0], X[1], X[2], s=0.1)
    ax.set_box_aspect([1, 1, 1])

    plotcams(P, ax)

def plotImage(filename, P, x):
    image = mpimg.imread("../assignment2data/" + filename)
    plt.imshow(image, cmap='gray')
    
    projX = pflat(np.matmul(P, X))
    plt.scatter(x[0], x[1], s=0.3, c='r')
    plt.scatter(projX[0], projX[1], s=0.3, c='g')

#for i in range(0, 9):
#    plotImage(imfiles[i][0], P[i], x[i])
#    plt.show()
    
plotReconstruction(X, P)
print(len(P[0]))

#for i in range(0, 9):
#    plotReconstruction(np.matmul(T1inv, X[i]),np.matmul(P[i], T1))
#    plt.show()

#plotReconstruction(pflat(np.matmul(T1inv, X)), P1)
#
#plt.show()
#
#
#plotReconstruction(pflat(np.matmul(T2inv, X)), P2)

#plt.show()