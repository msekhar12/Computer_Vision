#Module:1
#Chapter: 1.4 Basic image processing
#1.4.1: Translation

#The material presented in 1.4 (Basic image processing) is very important, although it looks quite basic.
#These concepts act as building blocks for more advanced CV algorithms and techniques

#Translation is the shifting of an image along the x and y axis.
#Using translation, we can shift an image up, down, left, or right, along with any combination of the above

#We will use a numpy matrix as translation matrix. It will have the following format:
#[[1, 0, shiftX], [0, 1, shiftY]]
#where:
#shiftx = The Number of pixels the image to be shifted along x-axis (can be positive or negative).
#If positive then the image will be shifted to right, else left. If 0, the image is NOT
#shifted along x-axis
#shifty = The Number of pixels the image to be shifted along y-axis (can be positive or negative).
#If positive then the image will be shifted to down, else up. If 0, the image is NOT
#shifted along y-axis

#We will use cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) function to perform the translation
#where image = image to be shifted
#      M = translation matrix
#      (image.shape[1], image.shape[0]) = (width, height)

#Example:
#The following code will shift the image 25 pixels right and 50 pixels down
#Then it will shift the image 50 pixels towards left and 90 towards up
#To execute: python Module_1_Lesson_1.4.1.py -i <image path>
#            something like: python Module_1_Lesson_1.4.1.py -i ../images/doc_gray.png
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

#Shift the image 25 pixels to right and 50 pixels down
#Our translation matrix M is defined as a floating point array.
#This is important because OpenCV expects this matrix to be of floating point type.
M = np.float32([[1,0,25],[0,1,50]])
shifted = cv2.warpAffine(image, M, (image.shape[1],image.shape[0]))

#Shifting the already shifted image 50 pixels towards left and 90 towards up
M = np.float32([[1,0,-50],[0,1,-90]])
re_shifted = cv2.warpAffine(shifted, M, (shifted.shape[1],shifted.shape[0]))

#Show the image:
cv2.imshow("original", image)
cv2.imshow("shifted", shifted)
cv2.imshow("re_shifted", re_shifted)
cv2.waitKey(0)
 
#NOTE:
#Once the image is shifted and if there is any loss in the image (chopped)
#then re-shift in the opposite direction will not re-claim the chopped part
 
#Manually constructing this translation matrix and calling the cv2.warpAffine
#method takes a fair amount of code — and it’s not pretty code either!
#Let us define a helper function to perform translation

import numpy as np
import cv2

def translate(image, x=0, y=0):
                # define the translation matrix and perform the translation
                M = np.float32([[1, 0, x], [0, 1, y]])
                shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
               # return the translated image
                return shifted

cv2.imshow("shifted_again",translate(image, x=-100))
cv2.waitKey(0)