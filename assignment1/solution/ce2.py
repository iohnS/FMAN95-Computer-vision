import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


image = mpimg.imread("../assignment1data/compEx2.JPG")

data = scipy.io.loadmat("../assignment1data/compEx2.mat")
p1 = data["p1"]
p2 = data["p2"]
p3 = data["p3"]

print(p1[0])

def plotPoint(point):
    plt.scatter(point[0], point[1])
    col = (np.random.random(), np.random.random(), np.random.random())
    plt.axline((point[0][0], point[1][0]), (point[0][1], point[1][1]), color=col)

plt.imshow(image, cmap='gray')
plotPoint(p1)
plotPoint(p2)
plotPoint(p3)
plt.show()

# Looking at the image the lines does not look parallel in 2D. But they are parallel in real life. They will converge in a point, being the vanish point
# Which is where p2 and p3 interscect. 