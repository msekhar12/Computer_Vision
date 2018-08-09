#Module:1
#Chapter: 1.11: Contours
#1.11.2: Simple contour properties
 
#In this lesson we will discuss the following properties of contours:
 
#1. Centroid/Center of Mass
#2. Area
#3. Perimeter
#4. Bounding boxes
#5. Rotated bounding boxes
#6. Minimum enclosing circles
#7. Fitting an ellipse
 
#NOTE to discuss these concepts, we need to understand "moments" of image ("image moments"). "Moments" are discussed in module 10.
#So for time being just understand that we have something called "Moments" for an image, and we will understand
#image moments in module 10
 
#1. Centroid/Center of Mass
#The centroid is simply the mean (i.e. average) position of all (x, y) coordinates along the contour of the shape.
 
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

# find external contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
clone = image.copy()

# loop over the contours
for c in cnts:
                # compute the moments of the contour which can be used to compute the
                # centroid or "center of mass" of the region
    # The cv2.moments() function returns a dictionary of moments with the keys of the dictionary
    # as the moment number and the values as the actual calculated moment.
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the center of the contour on the image
                cv2.circle(clone, (cX, cY), 10, (0, 255, 0), -1)

# show the output image
cv2.imshow("Centroids", clone)
cv2.waitKey(0)
clone = image.copy()
 
 
#2, 3 Area and Perimeter
#The area of the contour is the number of pixels that reside inside the contour outline.
#Similarly, the perimeter (sometimes called arc length) is the length of the contour.
 
#To compute area of a contour:
#area = cv2.contourArea(c)
#where c = contour
 
#To find perimeter of a contour:
#perimeter = cv2.arcLength(c, True)
#c = contour
#The second parameter indicates if the contour has NO gaps in the perimeter.
#True means NO gaps
 
 
#NOTE: To write text on the image, we can use cv2.putText() function:
#Example:
#font                   = cv2.FONT_HERSHEY_SIMPLEX
#bottomLeftCornerOfText = (10,500)
#fontScale              = 1
#fontColor              = (255,255,255)
#lineType               = 2
#cv2.putText(img,'Hello World!',
#    bottomLeftCornerOfText,
#    font,
#    fontScale,
#    fontColor,
#    lineType)
 
#Just like the contour area, we can use the perimeter as a method to quantify the shape of an object.
#However, the perimeter has a more important role to play when we explore contour approximation in a few sections.
 
#Example code:
# loop over the contours again
for (i, c) in enumerate(cnts):
                # compute the area and the perimeter of the contour
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(i + 1, area, perimeter))

                # draw the contour on the image
                cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
                # compute the center of the contour and draw the contour number
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                1.25, (255, 255, 255), 4)

# show the output image
cv2.imshow("Contours", clone)
cv2.waitKey(0)
 
 
#4. Bounding boxes
#A bounding box is an upright rectangle that “bounds” and “contains” the entire contoured region of the image.
#However, it does not consider the rotation of the shape, so you’ll want to keep that in mind.
#A bounding box consists of four components: the starting x-coordinate of the box, then the starting y-coordinate of the box,
#followed by the width and height of the box.
 
#To fit a bounding rectangle for a contour, we will use the following 2 statements:
#(x, y, w, h) = cv2.boundingRect(c)
#cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
#The statement (x, y, w, h) = cv2.boundingRect(c) will return the (x,y) and (w,h) of the bounding rectangle to
#enclose the contour "c"
 
#The statement #cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2) will draw a bounding rectangle
#with the supplied (x,y) and (x+w, y+h) parameters
 
#Example code:
# clone the original image
clone = image.copy()

# loop over the contours
for c in cnts:
                # fit a bounding box to the contour
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Bounding Boxes", clone)
cv2.waitKey(0)
clone = image.copy()
 
 
#5. Rotated bounding boxes
#To draw a rotated bounding box, which aligns to the orientation of the object, we cannot use the code
#given for bounding boxes
#The function cv2.minAreaRect(c) gives a tuple of 3 values:
#The first value is (x,y) coordinate of the rectangle
#The second value is the (w,h) (width and height)
#The third value is the angle of rotation needed to enclode the object.
#example: box = cv2.minAreaRect(c)
#We cannot use the output of cv2.minAreaRect(c) and draw a rectangle using cv2.rectangle() function,
#since cv2.rectangle() always draws a rectangle which is not rotated. Instead we will use cv2.boxPoints(box) to
#get the contours to be drawn. The obtained contour will be passed to cv2.drawContours(), to draw the rotated box.
 
#Example:
# loop over the contours
for c in cnts:
                # fit a rotated bounding box to the contour and draw a rotated bounding box
                box = cv2.minAreaRect(c)
    #For OpenCV 2.4, use cv2.cv.BoxPoints
                box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
                cv2.drawContours(clone, [box], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow("Rotated Bounding Boxes", clone)
cv2.waitKey(0)
clone = image.copy()
 
#In general, you’ll want to use standard bounding boxes when you want to crop a shape from an image.
#And you’ll want to use rotated bounding boxes when you are utilizing masks to extract regions from an image.
 
 
 
#6. Minimum enclosing circles
#Just as we can fit a rectangle to a contour, we can also fit a circle.
#The function: cv2.minEnclosingCircle() will take a contour as input and returns the
#enclosing circle's center (x,y) and the radius
#example: ((x, y), radius) = cv2.minEnclosingCircle(c)
 

for c in cnts:
                # fit a minimum enclosing circle to the contour
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)

# show the output image
cv2.imshow("Min-Enclosing Circles", clone)
cv2.waitKey(0)
clone = image.copy()
 
 
 
#7. Fitting an ellipse
#Fitting an ellipse to a contour is much like fitting a rotated rectangle to a contour.
#Under the hood, OpenCV is computing the rotated rectangle of the contour.
#And then it’s taking the rotated rectangle and computing an ellipse to fit in the rotated region.
#The function cv2.fitEllipse() will take a contour as input and returns the values needed to fit an ellipse.
#Then we can use the output of cv2.fitEllipse() to fit the ellipse using cv2.ellipse() function
 
#NOTE: Ellipse cannot fit a rectangle.
 
#Example:
#ellipse = cv2.fitEllipse(c)
#where c is contour
#cv2.ellipse(clone, ellipse, (0, 255, 0), 2)
 
# loop over the contours
for c in cnts:
                # to fit an ellipse, our contour must have at least 5 points
                if len(c) >= 5:
                                # fit an ellipse to the contour
                                ellipse = cv2.fitEllipse(c)
                                cv2.ellipse(clone, ellipse, (0, 255, 0), 2)

# show the output image
cv2.imshow("Ellipses", clone)
cv2.waitKey(0)
 
#Most of the times we use bounding boxes, and sometimes the rotated bounding boxes.
 
#Questions (the code to answer  the questions is given at the end):
#For this quiz, you’ll need to download the following image: http://pyimg.co/ye485
#(see the file shapes_example.png in images)
 
#Then use contour properties to answer “What is the centroid of the purple circle?”
#Answer: (148, 225). Used the following code to find the centroid
 
#What is the area of the circle?
#A: 9722
#Used the code given in this document (just execute, and see the output)
 
#What is the perimeter of the red square?
#A. 440
#Used the code given in this document (just execute, and see the output)
 
#What is the bounding box of the orange triangle?
#(177, 41, 110, 97)
 
#What is the radius of the minimum enclosing circle for the orange triangle?
#63
 
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
for (i, c) in enumerate(cnts):
                # compute the area and the perimeter of the contour
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(i + 1, area, perimeter))
    # draw the contour on the image
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
    # compute the center of the contour and draw the contour number
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 4)
    print("Contour #{} -- Center: {},{}".format(i + 1, cX, cY))
    (x, y, w, h) = cv2.boundingRect(c)
    print("Contour #{} -- Bounding Rectangle: ({},{},{},{})".format(i + 1, x, y, w, h))
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    print("Contour #{} -- Min enclosing circle: ({},{},{})".format(i + 1, x, y, radius))