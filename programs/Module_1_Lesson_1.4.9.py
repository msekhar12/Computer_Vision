#Module:1
#Chapter: 1.4 Basic image processing
#1.4.9: Splitting and Merging channels
#Image splitting

#A coloured image will have three channels: Red, Green, Blue.
#You can split these channels into individual matrices using cv2.split() function.
#You can merge the channels back using cv2.merge() function.

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

#The split will return the Blue, Green and Red channel matrices (in that order)
B, G, R = cv2.split(image)

#These matrices are 2D since there is no channel info
print(B.shape)
cv2.imshow("original",image)
cv2.imshow("Blue", B)
cv2.imshow("Green", G)
cv2.imshow("Red", R)

print("Image writing...")

cv2.imwrite("Red.jpg",R)

print("Image writing done...")

#we can merge the channels:

image_merged = cv2.merge([B, G, R])
cv2.imshow("merged", image_merged)

#DISPLAYING ONLY RED channel, and making other channels as 0
zeros = np.zeros(image.shape[0:2],dtype="uint8")

#Observe the order. For red, you need to use [zeros, zeros, R] since the first two channels should be zeros
cv2.imshow("red only",cv2.merge([zeros, zeros, R]))
cv2.imshow("green only",cv2.merge([zeros, G, zeros]))
cv2.imshow("blue only",cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)