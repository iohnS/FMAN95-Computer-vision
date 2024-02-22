import scipy
import numpy as np 
import matplotlib.pyplot as plt
import random

K = scipy.io.loadmat("../assignment3data/compEx3data.mat")['K']
Kinv = np.linalg.inv(K)
x = scipy.io.loadmat("../assignment3data/compEx1data.mat")['x']
[x1, x2] = [x[0][0], x[1][0]] # 2008 points.

normx1 = np.matmul(Kinv, x1)
normx2 = np.matmul(Kinv, x2)

n1t = np.transpose(normx1)
n2t = np.transpose(normx2)

M = []
for i in range(0, len(n1t)):
    row = [n1t[i][j] * n2t[i][jj] for j in range(0, len(n1t[0])) for jj in range(0, len(n2t[0]))]
    M.append(row)
    

[U, S, V] = np.linalg.svd(M)
solution = V[len(V)-1:][0]
Mv = np.linalg.norm(M @ solution) # Check so that Mv = 0 (close enough)
Eapprox = np.reshape(solution, [3, 3])
[U, S, V] = np.linalg.svd(Eapprox)
if np.linalg.det(U @ V) > 0:
    E = U @ np.diag([1,1,0]) @ V
else:
    V = [[-e for e in v]for v in V]
    E = E = U @ np.diag([1,1,0]) @ V

    
detE = np.linalg.det(E) #Should be zero, in this case it is close (8.74e⁻18)
#plt.plot(np.diag((np.transpose(normx2) @ E @ normx1))) # Should be close to zero which it is (±0.03)
F = np.transpose(Kinv) @ E @ Kinv

l = F @ x1
randomNbr = [random.randint(0, len(n1t)) for _ in range(0, 20)]
points = [np.transpose(x2)[i] for i in randomNbr]
ls = np.transpose(l)

def pyRital():
    # Have to loop as axline only takes a point + a slope
    for l in ls:
        plt.axline((l[0], l[1]), l[1]/l[0])

#plt.imshow(kronan2)
#plt.scatter([x[0] for x in points], [x[1] for x in points], c='r', s=10)
#pyRital()

xp1 = np.transpose(x1)
xp2 = np.transpose(x2)

def distance(i):
    x, y, z = xp2[i]
    a, b, c = ls[i]
    return np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

#distances = [distance(i) for i in range(0, len(xp2))]
#print(np.mean(distances))
#plt.hist(distances, bins=100)
#plt.show()
solE = [[el/E[2][2] for el in row] for row in E] # Fundamental matrix E with E/E[3,3]