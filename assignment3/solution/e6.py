import numpy as np
from sympy import *

var('s U W V Xs')

# Help for solving exercise 6 in assignment 3. 

U = [[1/np.sqrt(2), -1/np.sqrt(2), 0], [1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 1]]
V = [[1,0,0], [0,0,-1], [0, 1,0]]
W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

Xs = Matrix([0, 0, 1, s])

sol1 = np.hstack([np.matmul(np.matmul(U, W), np.transpose(V)), [[0], [0], [1]]])
sol2 = np.hstack([np.matmul(np.matmul(U, W), np.transpose(V)), [[0], [0], [-1]]])
sol3 = np.hstack([np.matmul(np.matmul(U, np.transpose(W)), np.transpose(V)), [[0], [0], [1]]])
sol4 = np.hstack([np.matmul(np.matmul(U, np.transpose(W)), np.transpose(V)), [[0], [0], [-1]]])

print(Matrix(sol4).multiply(Xs))
