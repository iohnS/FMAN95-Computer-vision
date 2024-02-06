import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pflat import pflat
from sympy import Matrix


data = sp.io.loadmat("../assignment2data/compEx3data.mat")                                              #Read data
Xmodel = data['Xmodel'] # point model. We have 37 3D points to work with. 
endind = data['endind']# used for plotting lines on the model surface.
startind = data['startind'] # used for plotting lines on the model surface.
x = data['x'][0] # measured PROEJCTIONS of the model points in the two images. 
cube1 = mpimg.imread("../assignment2data/cube1.JPG")
cube2 = mpimg.imread("../assignment2data/cube2.JPG")
[x1, x2] = [x[0], x[1]]

xy1_mean = [np.mean([x1[i]]) for i in range(0,2)]                                                   #Mean and standard deviation for the two images
xy2_mean = [np.mean([x2[i]]) for i in range(0,2)]
xy1_std = [np.std([x1[i]]) for i in range(0,2)]
xy2_std = [np.std([x2[i]]) for i in range(0,2)]

N1 = [[1/xy1_std[0], 0, -xy1_mean[0]/xy1_std[0]], [0, 1/xy1_std[1], -xy1_mean[1]/xy1_std[1]], [0,0,1]]  #Normalization matrices
N2 = [[1/xy2_std[0], 0, -xy2_mean[0]/xy2_std[0]], [0, 1/xy2_std[1], -xy2_mean[1]/xy2_std[1]], [0,0,1]]

norm1 = N1 @ x1                                                                                         #Normalized points                                          
norm2 = N2 @ x2

#plt.scatter(norm1[0], norm1[1])
#plt.scatter(norm2[0], norm2[1])

homX = list(Xmodel) + [[1 for _ in range(0, len(Xmodel[0]))]]

threeDPoints = np.transpose(homX)                                                 # Each row in the matrix is a 3D point
zeros = [.0 for _ in range(0, len(threeDPoints[0]))]                                       #Zeros for the M matrix with same length as the 3D points
zer = [0]

def createM(x):
    M = []
    for i in range(0, len(threeDPoints)):
        M.append(list(threeDPoints[i]) + zeros + zeros + zer*i + [-x[0][i]] + zer*(len(threeDPoints) - (i + 1)))
        M.append(zeros + list(threeDPoints[i]) + zeros + zer*i + [-x[1][i]] + zer*(len(threeDPoints) - (i + 1)))
        M.append(zeros + zeros + list(threeDPoints[i]) + zer*i + [float(-1)] + zer*(len(threeDPoints) - (i + 1)))
    return M 



ex =  [[3, 0, 0], [0, 2, 0], [0, 0, 1]]                                                      # Reshape is correct

M1 = createM(norm1)
[U1, S1, V1] = np.linalg.svd(M1)                                                    #svd returns Hermitian matrix for V. 
v1 = V1[len(V1)-1:][0]                                                       
M2 = createM(norm2)
[U2, S2, V2] = np.linalg.svd(M2)
v2 = V2[len(V2)-1:][0]
v2 = V2[len(V2)-1:][0]



P1 = np.linalg.inv(N1) @ np.reshape((v1[:12]), [3, 4])
P2 = np.linalg.inv(N2) @ np.reshape((v2[:12]), [3, 4])
P1np = np.array(Matrix(P1).nullspace()[0])      # The camera centers are negative so we have to multiply with -1
P2np = np.array(Matrix(P2).nullspace()[0])
cc1 = [e[0] for e in P1np][:3]
cc2 = [e[0] for e in P2np][:3]
pa1 = P1[2][0:2]
pa2 = P2[2][0:2]

#plt.quiver(cc1[0], cc1[1], pa1[0], pa1[1])
#plt.quiver(cc2[0], cc2[1], pa2[0], pa2[1])
        
x1im = pflat(np.matmul(P1, homX))
x2im = pflat(np.matmul(P2, homX))


#plt.imshow(cube2)
#for i in range(0, len(x2im[0])):
#    plt.scatter(x2[0][i], x2[1][i], c='b')
#    plt.scatter(x2im[0][i], x2im[1][i], c='r')

#plt.show()


K1 = np.linalg.qr(P1)[0]
K2 = np.linalg.qr(P2)[0]
