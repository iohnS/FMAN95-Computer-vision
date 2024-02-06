# SIFT: Scale-invariant Feature Transform

#https://github.com/rmislam/PythonSIFT  
#https://pypi.org/project/opencv-python/

import cv2 as cv

sift = cv.SIFT.create()

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

img1 = cv.imread("../assignment2data/cube1.JPG")
img2 = cv.imread("../assignment2data/cube2.JPG")

img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)


matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key = lambda x : x.distance)


x1 = [] # The matched points for image1
x2 = [] # Coordinates for the matched points in image2.

for m in matches:
    p1 = keypoints1[m.queryIdx].pt
    p2 = keypoints2[m.trainIdx].pt
    x1.append(p1)
    x2.append(p2)
    

img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[40:50], img2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#cv.imshow("Matches", img3)
#cv.waitKey(0)


