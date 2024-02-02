import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from pflat import pflat


data = scipy.io.loadmat("../assignment1data/compEx3.mat")
endpoints = data["endpoints"]
startpoints = data["startpoints"]

# The startingpoints and endpoints need to have ones padded to them to be able to use matrix multiplication wiuth the transform matrix. 

def print_grid(sp, ep, transform):
    startpoints = np.array(pflat(np.matmul(transform, sp)))
    endpoints = np.array(pflat(np.matmul(transform, ep)))
    
    for i in range(0, len(ep1[0])):
        plt.plot([startpoints[0][i], endpoints[0][i]], [startpoints[1][i], endpoints[1][i]], "b")

ones = [1 for _ in range(0, len(endpoints[0]))]
ep1 = np.append(endpoints, [ones], axis=0)
sp1 = np.append(startpoints, [ones], axis=0)

h0 = [[1,0,0], [0,1,0], [0,0,1]]
h1 = [[np.sqrt(3), -1, 1], [1, np.sqrt(3), 1], [0, 0, 2]]
h2 = [[1, -1, 1], [1, 1, 0], [0, 0, 1]]
h3 = [[1, 1, 0], [0,2,0], [0, 0, 1]]
h4 = [[np.sqrt(3), -1, 1], [1, np.sqrt(3), 1], [1/4, 1/2, 2]]

print_grid(sp1, ep1, h1)

plt.show()