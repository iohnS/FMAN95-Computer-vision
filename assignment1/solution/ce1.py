import scipy.io
import matplotlib.pyplot as plt
from pflat import pflat

data = scipy.io.loadmat("../assignment1data/compEx1.mat")
x2D = data["x2D"]
x3D = data["x3D"]

# To get the last entry of points from the x2D. 
# We want to divide each column with the last row of the same column. 


pflat2d = pflat(x2D)
plt.scatter(pflat2d[0], pflat2d[1])
plt.show()

pflat3d = pflat(x3D)
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(pflat3d[0], pflat3d[1], pflat3d[2])

plt.show()