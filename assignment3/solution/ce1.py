import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
from pflat import pflat
from rital import rital


x = scipy.io.loadmat("../assignment3data/compEx1data.mat")['x']
[x1, x2] = [x[0][0], x[1][0]] # 2008 points.
kronan1 = mpimg.imread("../assignment3data/kronan1.JPG")
kronan2 = mpimg.imread("../assignment3data/kronan2.JPG")

xy1_mean = [np.mean([x1[i]]) for i in range(0,2)]                                                   #Mean and standard deviation for the two images
xy2_mean = [np.mean([x2[i]]) for i in range(0,2)]
xy1_std = [np.std([x1[i]]) for i in range(0,2)]
xy2_std = [np.std([x2[i]]) for i in range(0,2)]

N1 = [[1/xy1_std[0], 0, -xy1_mean[0]/xy1_std[0]], [0, 1/xy1_std[1], -xy1_mean[1]/xy1_std[1]], [0,0,1]]  #Normalization matrices
N2 = [[1/xy2_std[0], 0, -xy2_mean[0]/xy2_std[0]], [0, 1/xy2_std[1], -xy2_mean[1]/xy2_std[1]], [0,0,1]]


norm1 = N1 @ x1                                                                                         #Normalized points                                          
norm2 = N2 @ x2
n1t = np.transpose(norm1)
n2t = np.transpose(norm2)

M = []

for i in range(0, len(n1t)):
    row = [n1t[i][j] * n2t[i][jj] for j in range(0, len(n1t[0])) for jj in range(0, len(n2t[0]))]
    M.append(row)
    
[U, S, V] = np.linalg.svd(M)
solution = V[len(V)-1:][0]
Fe = np.reshape(solution, [3, 3])   #Fundamental matrix estimation.
detF = np.linalg.det(Fe) 
[U, S, V] = np.linalg.svd(Fe)
S[-1] = 0
F = U @ np.diag(S) @ V
detF = np.linalg.det(F)

Mv = np.linalg.norm(F @ V[len(V)-1:][0])
epiConstr = np.diag(np.transpose(norm2) @ F @ norm1)
nF = np.transpose(N2) @ F @ N1

l = F @ x1
l1 = l[0]
l2 = l[1]
sum_squares = l1**2 + l2**2
sum_squares_replicated = np.tile(sum_squares, (3, 1))
sqrt_sum_squares = np.sqrt(sum_squares_replicated)
l = l / sqrt_sum_squares

xp2 = np.transpose(x2)
xp1 = np.transpose(x1)
ls = np.transpose(l)
randomNbr = [random.randint(0, len(n1t)) for _ in range(0, 20)]
points = [np.transpose(x2)[i] for i in randomNbr]

def pyRital():
    # Have to loop as axline only takes a point + a slope
    for l in ls:
        plt.axline((l[0], l[1]), l[1]/l[0])

#plt.imshow(kronan2)
#plt.scatter([x[0] for x in points], [x[1] for x in points], c='r', s=10)
#pyRital()

def distance(i):
    x, y, z = xp2[i]
    a, b, c = ls[i]
    return np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

distances = [distance(i) for i in range(0, len(xp2))]
#print(np.mean(distances))
#plt.hist(distances, bins=100)
#plt.show()
 
#solF = [[el/F[2][2] for el in row] for row in F] # Fundamental matrix F with F/F[3,3]
