#Module:1
#Chapter: 1.4 Basic image processing
#1.4.8: Masking

#We can use a combination of both bitwise operations and masks to construct ROIs that are non-rectangular. 
#This allows us to extract regions from images that are of completely arbitrary shape.

#A mask allows us to focus only on the portions of the image that interests us.

#For example, let’s say that we were building a computer vision system to recognize faces. 
#The only part of the image we are interested in finding and describing are the parts of the image 
#that contain faces — we simply don’t care about the rest of the content of the image. 
#Provided that we could find the faces in the image, we might construct a mask to show only the faces in the image.


#In general we will use Machine Learning algorithms to automatically detect object(s) of interest in an image 
#and extract them. But in this lesson we will demonstrate the masking assuming that we know the exact location of
#our ROI.

#In masking, we will define a canvas with the required shape filled with white color or 255 pixel value
#Then we will combine the images by bitwise AND to extract the image present at the white pixels

#This program is written to extract the dinosaur's face in trex.png image using masking technique.
#So use the following command to execute the program on trex.png (assuming that you downloaded the image 
#to images directory in the directory where this program is saved ):
#python Module_1_Lesson_1.4.8.py -i ./images/trex.png

#NOTE: The mask will be a 2D matrix (which is a binary image), while the image on which the mask is applied 
#can be a coloured image with coloured channels. 


import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i","--image",required=True, help = "Path to image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print(image.shape)
#350, 228
#234:, 0:124

#define a rectangular image:
#In this code, I supplied the rectangle coordinates, by estimating the location of dinosaur's face
#The goal is to mask everything, except the dinosaur's face
#Observe that the depth of the image is not supplied, as the mask will be a black and white picture anyway

rectangle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.rectangle(rectangle, (224, 40),(325, 124), 255, -1)

cv2.imshow("original",image)
cv2.imshow("rectangle",rectangle)

#Observe the syntax. The "mask=rectangle" is supplied as input
cv2.imshow("masked_rect",cv2.bitwise_and(image, image, mask=rectangle))

circle = np.zeros(image.shape[0:2] ,dtype="uint8")
cv2.circle(circle, (280, 80), 50, 255, -1)
cv2.imshow("masked_circle",cv2.bitwise_and(image, image, mask=circle))

cv2.waitKey(0)
