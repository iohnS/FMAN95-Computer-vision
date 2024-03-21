from triangulate import triangulate
import numpy as np 
import matplotlib.pyplot as plt
from sympy import Matrix
from pflat import pflat


def show3d(P1, P2, x1, x2):
    P1np = np.array(Matrix(P1).nullspace()[0])
    P2np = np.array(Matrix(P2).nullspace()[0])
    cc1 = [e[0] for e in P1np]
    cc2 = [e[0] for e in P2np]
    pa1 = P1[2][0:3]
    pa2 = P2[2][0:3]
    
    X = pflat(np.transpose(triangulate(P1, P2, x1, x2)))
    
    fig3d = plt.figure()
    sp3d = fig3d.add_subplot(projection='3d')
    sp3d.scatter(cc1[0], cc1[1], cc1[2], c='r')
    sp3d.scatter(cc2[0], cc2[1], cc2[2], c='r')
    sp3d.quiver(cc1[0], cc1[1], cc1[2], pa1[0], pa1[1], pa1[2], color="g")
    sp3d.quiver(cc2[0], cc2[1], cc2[2], pa2[0], pa2[1], pa2[2], color="g")
    sp3d.scatter(X[0], X[1], X[2], c='b', s=0.5)
    sp3d.set_xlabel('X')
    sp3d.set_ylabel('Y')
    sp3d.set_zlabel('Z')
    plt.show()
    
    
def getBest(Ps, x1, x2):
    P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]  
    bestP = Ps[0]
    inFront = 0
    bestX = []
    for P2 in Ps:
        current = 0
        X = triangulate(P1, P2, x1, x2)
        for p in X:
            if p[2] > 0:
                current += 1
        if current > inFront:
            inFront = current
            bestX = X
            bestP = P2
        print(current)
    return [bestP, bestX]
