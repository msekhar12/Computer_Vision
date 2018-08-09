#Module:1
#Chapter: 1.11: Contours
#1.11.5: Sorting contours
 
#The topic of contour sorting extremely useful and applicable to a variety of image processing projects.
 
#For instance in license plate recognition we need to sort the contours to identify the text in correct order.
 
#For CV projects sometimes we need to sort the contours based on the following criteria:
#1. size/area (see the lesson "1.11.4: Contour approximation" or example statement below)
#2. left-to-right
#3. right-to-left
#4. top-to-bottom
#5. bottom-to-top
 
#Sorting contours based on the size is easy. see below example statement:
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
 
#Let us define a function to sort contours:
#This function will be our helper function to sort the contours, based on the
#position of the contours (2, 3, 4, 5 listed above)
 
# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
 
#In the following function:
#cnts = list of contours we want to sort
#method = "left-to-right" (default)
#       = "right-to-left"
#       = "top-to-bottom"
#       = "bottom-to-top"
 

def sort_contours(cnts, method="left-to-right"):
                # initialize the reverse flag and sort index
                reverse = False
                i = 0

                # handle if we need to sort in reverse
                if method == "right-to-left" or method == "bottom-to-top":
                                reverse = True

                # handle if we are sorting against the y-coordinate rather than
                # the x-coordinate of the bounding box
                if method == "top-to-bottom" or method == "bottom-to-top":
                                i = 1

                # construct the list of bounding boxes and sort them from top to
                # bottom
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                key=lambda b:b[1][i], reverse=reverse))

                # return the list of sorted contours and bounding boxes
                return (cnts, boundingBoxes)


#Let us define another helper function to label contours
def draw_contour(image, c, i):
                # compute the center of the contour area and draw a circle
                # representing the center
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw the countour number on the image
                cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 255, 255), 2)

                # return the image with the contour number drawn on it
                return image

   

#Read the lesson for an example            