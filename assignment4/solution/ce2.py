import numpy as np
import scipy.io
import matplotlib.image as mpimg

data = scipy.io.loadmat("../assignment4data/compEx2data.mat")
K = data['K']
[x1, x2] = data['x'][0]

im1 = mpimg.imread("../assignment4data/im1.jpg")
im2 = mpimg.imread("../assignment4data/im2.jpg")
