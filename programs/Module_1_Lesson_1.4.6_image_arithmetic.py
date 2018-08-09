#Module:1
#Chapter: 1.4 Basic image processing
#1.4.6: Image arithmetic

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

#In images, the pixels can take a value of [0, 255].
#This is the reason to represent the pixels as unsigned small int, with 1 byte size.
#Any value out of this range must be either restricted or wrapped around.
#For example if we add 10 to 250, then either we need to restrict the resulting value to 255 or 4 (wrapped around)
#On images, we can perform numpy arithmetic or cv2 arithmetic.
#Numpy arithmetic will wrap around the values, while cv2 arithmetic will restrict the values between [0, 255]
#Which one to use depends on the requirement.

#Image arithmetic can be performed in 2 ways
#Add 10 to 250 using cv2 arithmetic:
print(cv2.add(np.uint8([10]), np.uint8([250])))
#will print 255

#where as:
print(np.uint8([10]) + np.uint8([250]))
#will print [4]

#subtract 100 from 50 using cv2
print(cv2.subtract(np.uint8([50]), np.uint8([100])))
#will show [[0]]

#subtract 100 from 50 using numpy
print(np.uint8([50]) - np.uint8([100]))
#will show [206]

#Let us increase all the pixel values by 100
M = np.ones(image.shape, dtype="uint8") * 100

added = cv2.add(image, M)

#Let us decrease all the pixel values by 100
M = np.ones(image.shape, dtype="uint8") * 100

subtracted = cv2.subtract(image, M)

cv2.imshow("original",image)
cv2.imshow("added", added)
cv2.imshow("subtracted", subtracted)

#We can observe that as we increase the pixel values, the image gets much lighter,
#as as we decrease the pixel values, the image gets much darker

cv2.waitKey(0)

#Download the source code from this lesson. Add value of 75 to all pixels to the grand_canyon.png image using the cv2.add function. What is the value of the pixel located at x=61, y=152?
cv2.imshow("image", image)
print(cv2.add(image[152,61], 75))
cv2.waitKey(0)