#Module:1
#Chapter: 1.4 Basic image processing
#1.4.5: Cropping

#Cropping is the act of selecting and extracting the Region of Interest (or simply, ROI),
#which is the part of the image we are interested in.

#In a face detection application, we would want to crop the face from an image.
#And if we were developing a Python script to recognize dogs in images, we may want to
#crop the dog from the image once we have found it.

#When we crop an image, we want to remove the outer parts of the image that we are not interested in.
#This is commonly called selecting our Region of Interest, or more simply, our ROI.

import numpy
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

#To crop an image, it is straight forward...
#You just need to extract the required numpy region.

#For instance if you want to extract the rows 20 to 30 and columns 10 to 50 from a numpy array,
#all you need to do is:
#a[20:31,10:51], where "a" is numpy array

#But in cv2 world, the image has origin at the top right corner, and the x axis begins at origin and increases towards right (horizontal axis)
#The vertical axis begis at origin and as we move down it increases (vertical axis)
#so if you want to get the image between x1 = 10 and x2 = 50, and y1 = 20 and y2 = 30 then you need to convert it to numpy coordinates:
#image[y1:y2, x1:x2]

x1 = 10
x2 = 50
y1 = 20
y2 = 30

cropped = image[y1:y2,x1:x2]
cv2.imshow("original image", image)
cv2.imshow("cropped image", cropped)

#util function
def crop(image, x1=None, x2 = None, y1 = None, y2 = None):
    #x1, x2 are x values on the x-axis of the image
    #y1, y2 are y values on the y-axis of the image
    if x1 is None or x2 is None or y1 is None or y2 is None:
       return image
    return image[y1:y2,x1:x2]

cropped = crop(image,x1=100, x2=200, y1=10, y2=150)
cv2.imshow("re-cropped image", cropped)
cv2.waitKey(0)

#Download the source code and image to this lesson. Then, use the florida_trip.png image to answer the following question.
#What is the most appropriate NumPy array slice to crop the people in the background from the florida_trip.png image?
#image[85:250, 85:220]
#image[173:235, 13:81]
#image[124:212, 225:380]
#image[90:450, 0:290]

cv2.imshow("image-1", image[85:250, 85:220])
cv2.waitKey(0)
cv2.imshow("image-2", image[173:235, 13:81])
cv2.waitKey(0)
cv2.imshow("image-3", image[124:212, 225:380])
cv2.waitKey(0)
cv2.imshow("image-4", image[90:450, 0:290])
cv2.waitKey(0)

#Use the same image from the previous question and determine the best NumPy slice to extract the boat/building-like structure from the background of florida_trip.png.
#image[124:212, 225:380]