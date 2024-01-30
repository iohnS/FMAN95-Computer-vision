from matplotlib.lines import Line2D
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sympy import Matrix
import math


image = mpimg.imread("../assignment1data/compEx2.JPG")

data = scipy.io.loadmat("../assignment1data/compEx2.mat")
p1 = data["p1"]
p2 = data["p2"]
p3 = data["p3"]

def plotPoint(point):
    plt.scatter(point[0], point[1])
    col = (np.random.random(), np.random.random(), np.random.random())
    return plt.axline((point[0][0], point[1][0]), (point[0][1], point[1][1]), color=col)

plt.imshow(image, cmap='gray')
plotPoint(p1)
line2 = plotPoint(p2)
line3 = plotPoint(p3)

def find_intersection(point1, point2, point3, point4):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4

    
    slope1 = (y2 - y1) / (x2 - x1)
    intercept1 = y1 - slope1 * x1

    slope2 = (y4 - y3) / (x4 - x3)
    intercept2 = y3 - slope2 * x3


    x_intersection = (intercept2 - intercept1) / (slope1 - slope2)

    
    y_intersection = slope1 * x_intersection + intercept1

    return x_intersection, y_intersection

# Intersect P2 och P3
n1, n2 = zip(p2[0], p2[1])
n3, n4 = zip(p3[0], p3[1])
intersect_point = find_intersection(n1, n2, n3, n4)

plt.scatter(intersect_point[0], intersect_point[1], zorder=10)

# solve the linear equation system to find the line passing through p1. 
nullsp = Matrix(np.transpose(p1)).nullspace()[0]
d = np.abs(intersect_point[0] * nullsp[0] + intersect_point[1] * nullsp[1] + nullsp[2])/math.sqrt(nullsp[0]**2 + nullsp[1]**2)

print(d)

plt.show()