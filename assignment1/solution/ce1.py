import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat("../assignment1data/compEx1.mat")
x2D = data["x2D"]
x3D = data["x3D"]

def elementWiseDivision(l1, l2):
    newList = []
    if len(l1) == len(l2):
        for i in range(0, len(l1)):
            newList.append(l1[i] / l2[1])
    return newList

# To get the last entry of points from the x2D. 
# We want to divide each column with the last row of the same column. 
def pflat(data):
    newData = []
    lastCoordinates = data[len(data)-1]
    for row in data:
        print("NEW", row, lastCoordinates, elementWiseDivision(row, lastCoordinates))
        newData.append(elementWiseDivision(row, lastCoordinates))
    return newData

pflat2d = pflat(x2D)
plt.scatter(pflat2d[0], pflat2d[1])
plt.show()

pflat3d = pflat(x3D)
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(pflat3d[0], pflat3d[1], pflat3d[2])

plt.show()