#Module:1
#Chapter: 1.11: Contours
#1.11.1 Finding and Drawing Contours
 
 
#After using thresholding and edge detection, we were able to find the outlines of the objects of interst in the image.
#To extract those objects from the image, we need "contours"
#"Contours" is a very important topic in CV.
 
#Being able to leverage simple contour properties enables you to solve complicated problems with ease
#Too often computer vision developers try to leverage complicated Machine Learning techniques to solve problems where contours would be better suited
 
#"Contours" are simply the outlines of an object in an image
#If the image is simple enough, we might be able to get away with using the grayscale image as input.
#For complex images we need to perform activities such as blurring, gradients and/or edge detectors, binarize the image before applying the contours function
 
#In OpenCV, the following function can be applied to automatically detect the contours:
 
#cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#Where gray.copy() will make a copy of the image named "gray"
#The function cv2.findContours() is a destructive function and hence we need to supply
#a copy of the image as input, since the input image will be modified by the findContours() function.
#cv2.RETR_LIST will retrieve all the contours found on the image
#cv2.CHAIN_APPROX_SIMPLE This flag will return only the end points of contours.
#The other flag cv2.CHAIN_APPROX_NONE (opposite to cv2.CHAIN_APPROX_SIMPLE) will get all the (x,y) values of
#all the points on the contours. Using cv2.CHAIN_APPROX_NONE is not advisable, as it will substantially slower and requires significantly more memory
 
#In OpenCV3:
#The cv2.findContours function returns a tuple of values. The first value is the image itself.
#The 2nd value is the contours themselves. These contours are simply the boundary points of the outline along the object.
#The third value is the hierarchy of the contours, which contains information on the topology of the contours.
#Often we are only interested in the contours themselves and not their actual hierarchy (i.e. one contour being contained in another)
 
#In OpenCV 2.4:
#The cv2.findContours function returns a tuple of 2 values.
#The first value is contours.
#The second value is contours hierarchy
 
#To draw contours we will use the function cv2.drawContours()
#Example: cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
#The first is the image on which we would like to draw the contours.
#The second parameter is the contours list found via cv2.findContours() function
#The third parameter is the index of the contour inside the cnts  list that we want to draw.
#If we wanted to draw only the first contour, we could pass in a value of 0.
#If we wanted to draw only the second contour, we would supply a value of 1.
#Passing in a value of -1 for this argument instructs the cv2.drawContours  function to draw all contours in the cnts  list
#Finally, the last two arguments to the cv2.drawContours  function is the color of the contour (in this case green),
#and the thickness of the contour line (2 pixels).
 
# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")

args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original image
cv2.imshow("Original", image)

# find all contours in the image and draw ALL contours on the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("Found {} contours".format(len(cnts)))

# show the output image
cv2.imshow("All Contours", clone)
cv2.waitKey(0)

#How to access each individual contour:
# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()
 
# loop over the contours individually and draw each of them
for (i, c) in enumerate(cnts):
                print("Drawing contour #{}".format(i + 1))
                cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
                #cv2.imshow("Single Contour", clone)
                #cv2.waitKey(0)
 
#If I want to draw the first 2 contours, I can use the following code:
#cv2.drawContours(clone, cnts[:2], -1, (0, 255, 0), 2)
   
#If I want to draw the first 1 contour only, I can use the following code:
#cv2.drawContours(clone, [cnts[0]], -1, (0, 255, 0), 2)
 
#External contours
#If I do not want to extract all the contours, but extract only those contours which are external, and are not enclosed in any other contour,
#then I have to use the cv2.RETR_EXTERNAL as shown in the below function call:
_,cnts,_ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#Using both contours and masks together.
#Once contours are identified, we can use them to prepare mask, which includes only the contours.
#This mask is bitwise anded with the original image to extract the objects in the image
 
#Create a mask:
mask = np.zeros(gray.shape, dtype="uint8")
mask = cv2.drawContours(mask, cnts, -1, 255, -1)
 
#Show the mask:
cv2.imshow("Mask", mask)
 
#Bitwise and the mask with the image:
cv2.imshow("Image + Mask", cv2.bitwise_and(image, image, mask=mask))
 
cv2.waitKey(0)

#Questions 
#The cv2.findContours function is usually applied to images in which color space?
#Binary
 
#All of the following are a valid way to draw the 3rd contour in a list, except:
#A. cv2.drawContours(image, cnts, 2, (0, 255, 0), 1)
#B. cv2.drawContours(image, cnts[2], -1, (0, 255, 0), 1)
#C. cv2.drawContours(image, [cnts[2]], -1, (0, 255, 0), 1)
#Ans: B. The contours must be supplied as a list, even though you intend to access only one contour
#        The cnts[2] will be the thirs contour, and this is not a list.