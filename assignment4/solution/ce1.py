import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
from pflat import pflat
import math

# Homography need 4 samples. 
# 1. Take 4 random pointsA
# 2. Compute the homography using points
# 3. Count the number of inliers within the error (5 pixels)
# 4. Repeat 1-3 for N times
# 5. Choose the homography with the most inliers

# Assume that the points are already homogenous. 

def HM(pointsA, pointsB):
    zeros = [0 for _ in p1[0]]
    zer = [0]
    M = []
    for i in range(0, len(pointsA)):
        M.append(list(pointsA[i]) + zeros + zeros + zer*i + [-pointsB[i][0]] + zer*(len(pointsA) - (i + 1)))
        M.append(zeros + list(pointsA[i]) + zeros + zer*i + [-pointsB[i][1]] + zer*(len(pointsA) - (i + 1)))
        M.append(zeros + zeros + list(pointsA[i]) + zer*i + [-1] + zer*(len(pointsA) - (i + 1)))    
    return M


sift = cv.SIFT.create()

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

a = mpimg.imread("../assignment4data/a.jpg")
b = mpimg.imread("../assignment4data/b.jpg")

img1 = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(b, cv.COLOR_BGR2GRAY)

keypointsA1, descriptors1 = sift.detectAndCompute(img1, None)
keypointsA2, descriptors2 = sift.detectAndCompute(img2, None)


matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key = lambda x : x.distance)


xA = [] # The matched pointsA for image1
xB = [] # Coordinates for the matched pointsA in image2.

for m in matches:
    p1 = keypointsA1[m.queryIdx].pt
    p1 += (1,)
    p2 = keypointsA2[m.trainIdx].pt
    p2 += (1,)
    xA.append(p1)
    xB.append(p2)

N = 100 # The number of iterations for RANSAC
e = 5
bestInliers = 0
bestH = []
for i in range(0, N):
    p1 = []
    p2 = []
    for i in range(0, 4):
        idx = np.random.randint(0, len(matches))
        p1.append(xA[idx])
        p2.append(xB[idx])
    
    M = HM(p1, p2)
    U, S, V = np.linalg.svd(M)
    solution = V[-1]
    H = np.reshape(solution[:9], (3, 3))
    H = [[e/H[-1][-1] for e in row] for row in H]
    Hx = np.transpose(pflat(H @ np.transpose(xA)))
    inliers = 0
    for i in range(0, len(Hx)):
        if math.dist(Hx[i], xB[i]) <= e:
            inliers += 1
    if(inliers > bestInliers):
        bestInliers = inliers
        bestH = H

print(bestInliers)
print(bestH)

print(bestH @ np.transpose(xA))
print(bestH @ np.transpose(xB))

img3 = cv.drawMatches(img1, keypointsA1, img2, keypointsA2, matches, img2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("Matches", img3)
cv.waitKey(0)

