#Module:1
#Chapter: 1.4 Basic image processing
#1.4.2: Rotation

#The material presented in 1.4 (Basic image processing) is very important, although it looks quite basic.
#These concepts act as building blocks for more advanced CV algorithms and techniques

#Rotation deals with rotating an image using an arbitrary center.
#Assume that the center of the image is the origin, then each of the
#x and y values can be used to represent a vector: [x,y]
#This vector will be multiplied with [[cos(theta), -sin(theta)],[sin(theta), cos(theta)]]
#to rotate [x,y] by an angle of theta. A positive value of theta will rotate the vector in
#anti-clock direction (or counter clock wise), and a negative theta will rotate in clock wise direction.
#For example the vector [1,2] (x=1 and y = 2) if rotated by 45 deg counter clock wize will become:
#[cos(-45) + 2sin(-45), -sin(-45) +2 cos(-45)]

#The in-built cv2 rotation will also scale the image, if desired.
#See the gurus notes (1.4.2) to get the mathematical details

#Example:
#The following code will rotate the image by 45 degrees counter clock wise (with scale of 1 or no change in size)
#and 30 degrees clock wise with scale of 1.5 (increase by 50%).
#We will also rotate counter-clockwise by 180 deg with scale of .4 (40% or reduce by 60%)
#The center of the image will be used for rotation.
#To execute: python Module_1_Lesson_1.4.2.py -i <image path>
#            something like: python Module_1_Lesson_1.4.2.py -i ../images/doc_gray.png
#Import packages:

import cv2
import numpy as np
import argparse

#Create a command line argument to accept image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")

#Read the input arguments
args = vars(ap.parse_args())

#Read the input image
image = cv2.imread(args['image'])


#get the image's width and height
(h, w) = image.shape[:2]

#Define the center of the image:
(cX, cY) = (w/2,h/2)

#Define translation matrix.
#For translation we supplied our own translation matrix, since it was straight forward.
#But for rotation the translation matrix will be complex to compute.
#So we will use in-built function cv2.getRotationMatrix() to create the translation matrix for rotation
#45 as second parm specifies that the image will be rotated by 45 deg counter clockwise
#1 as third parm specifies the scaling.

M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)


#Perform the actual rotation
rotate_45_anti = cv2.warpAffine(image, M, (w,h))
cv2.imshow("rotate_45_anti",rotate_45_anti)
cv2.waitKey(0)

#Rotate 30 deg clock-wise, with 1.5 scale, using the center of the image
M = cv2.getRotationMatrix2D((cX, cY), -30, 1.5)
rotate_30_clock = cv2.warpAffine(image, M, (w,h))
cv2.imshow("rotate_30_clock",rotate_30_clock)
cv2.waitKey(0)

#Rotate with a different center (50 pixels towards left from center and 50 pixels towards top)
M = cv2.getRotationMatrix2D((cX-50, cY-50), -30, .5)
rotate_30_clock_diff_center = cv2.warpAffine(image, M, (w,h))
cv2.imshow("rotate_30_clock_diff_center",rotate_30_clock_diff_center)
cv2.waitKey(0)

#NOTE:
#Once the image is rotated and if there is any loss in the image (chopped)
#then opposite direction rotation will not re-claim the chopped part

#Manually constructing this translation matrix and calling the cv2.warpAffine
#method takes a fair amount of code — and it’s not pretty code either!
#Let us define a helper function to perform translation

 

import numpy as np
import cv2

def rotate(image, angle, center=None, scale=1.0):
                # grab the dimensions of the image
                (h, w) = image.shape[:2]
                # if the center is None, initialize it as the center of
                # the image
                if center is None:
                                center = (w / 2, h / 2)
                # perform the rotation
                M = cv2.getRotationMatrix2D(center, angle, scale)
                rotated = cv2.warpAffine(image, M, (w, h))
                # return the rotated image
                return rotated

cv2.imshow("rotated_again",rotate(image, angle=180,scale=.4))
cv2.waitKey(0)

#Q. Download the following image: http://pyimg.co/kwy7l
#Then, use OpenCV to rotate the image 30 degrees clockwise. What is the value of the pixel located at x=335 and y=254?

wynn_rotated = rotate(image, -30)
cv2.imshow("wynn_rotated",wynn_rotated)
cv2.waitKey(0)
print(wynn_rotated[254,335])
#answer: R=95, G=93, B=61 (OpenCV 3.3)
#Now rotate the image 110 degrees counter-clockwise. What is the value of the pixel located at x=312, y=136?
wynn_rotated = rotate(image, 110)
cv2.imshow("wynn_rotated",wynn_rotated)
cv2.waitKey(0)
print(wynn_rotated[136,312])

 

#Change the call to cv2.getRotationMatrix2D to rotate the image 88 degrees counter-clockwise about coordinate (50, 50). What is the value of the pixel located at point (10, 10)?
wynn_rotated = rotate(image, 88, center=(50, 50))
cv2.imshow("wynn_rotated",wynn_rotated)
cv2.waitKey(0)
print(wynn_rotated[10,10])