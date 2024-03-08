from LinearizeReprojErr import LinearizeReprojErr
from ComputeReprojectionError import ComputeReprojectionError
from update_solution import update_solution
import scipy.io
import numpy as np
from ce2 import P2, X
from pflat import pflat

# Implement the steepest descent algorithm with my own strategy of finding gamma. 
# Use the solution from computer exercise 2 (unnormalized cameras and 3D points).
# Might need a very small gamma to converge. 

data = scipy.io.loadmat("../assignment4data/compEx2data.mat")
[x1, x2] = data['x'][0]

P = P2 # The camera matrix. 3x4 
U = X # The 3D points in homogenous coordinates. 4xN
u = pflat(np.matmul(P, U)) # The image points in homogenous coordinates. 3xN


gamma = 0 # We want to find gamma so that the objective function decreases by making lambda small enough. 

[err, res] = ComputeReprojectionError(P, U, u) # Returns the total reproduction error and the error for each visible points. 
r, J = LinearizeReprojErr(P, U, u) # Linearizes the reprojection error. 

deltav = -gamma @ np.transpose(J) @ r
[Pnew, Unew] = update_solution(deltav, P, U) # Computes a new set of cameras and 3D points from an update deltav. 


