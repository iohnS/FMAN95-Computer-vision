import scipy.io
import matplotlib

data = scipy.io.loadmat("./assignment1data/compEx1.mat")
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
        newData.append(elementWiseDivision(row, lastCoordinates))
    return newData

print("kjashdf")

pflat(x2D)