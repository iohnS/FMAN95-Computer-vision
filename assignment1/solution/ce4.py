import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sympy import Matrix
from pflat import pflat
import multiprocessing as mp
import math

image1 = mpimg.imread("../assignment1data/compEx4im1.JPG")
image2 = mpimg.imread("../assignment1data/compEx4im2.JPG")

data = scipy.io.loadmat("../assignment1data/compEx4.mat")
K = data['K']
R1 = np.array(data['R1'])
t1 = data['t1']
R2 = data['R2']
t2 = data['t2']
U = data['U']

# Image1 = P1 = K[R1 t1]
# Image2 = P2 = K[R2 t2]
    
P1 = np.matmul(K, np.insert(R1, [3], [[t1[i][0]] for i in range(0, len(t1))], 1))
P2 = np.matmul(K, np.insert(R2, [3], [[t2[i][0]] for i in range(0, len(t2))], 1))


# Print P1, P2 and P1np, P2np to get the camera centers + principal axes
P1np = np.array(Matrix(P1).nullspace()[0])
P2np = np.array(Matrix(P2).nullspace()[0])


fU = pflat(U)

U1 = pflat(np.matmul(P1, U))
U2 = pflat(np.matmul(P2, U))

def plotp(coord):
    [xl, yl] = coord
    for i in range(0, len(xl)):
        print(i)
        plt.scatter(xl[i], yl[i], s=0.1, c='b')

#plt.imshow(image2, cmap='gray')
#plotp((U2[0], U2[1]))


def plot3d():
    cc1 = [e[0] for e in P1np]
    cc2 = [e[0] for e in P2np][:3]
    pa1 = P1[2][0:3]
    pa2 = P2[2][0:3]
    print(P1)
    
    fig3d = plt.figure()
    sp3d = fig3d.add_subplot(projection='3d')
    sp3d.scatter(cc1[0], cc1[1], cc1[2], c='r')
    sp3d.scatter(cc2[0], cc2[1], cc2[2], c='r')
    sp3d.quiver(cc1[0], cc1[1], cc1[2], pa1[0], pa1[1], pa1[2])
    sp3d.quiver(cc2[0], cc2[1], cc2[2], pa2[0], pa2[1], pa2[2])
    sp3d.set_xlabel("X")
    sp3d.set_ylabel("y")
    for i in range(0, len(fU[0])):
        print(i)
        sp3d.scatter(fU[0][i], fU[1][i], fU[2][i], s=0.15, c='b')
    
    plt.show()
plot3d()    

