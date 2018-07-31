#Module:1
#Chapter:1.3
#1.2: Drawing

import cv2
import numpy as np
import argparse

#Drawing by itself might not seem too exciting, but it’s important that you understand these techniques.
#While drawing by itself doesn’t allow us to visually understand the contents of an image using OpenCV,
#these operations allow us to draw Regions of Interest (ROIs) surrounding objects in an image.

#Create a canvas
canvas = np.zeros((300,300,3),dtype="uint8")
print("Shape of canvas")
print(canvas.shape)

cv2.imshow("canvas",canvas)
cv2.waitKey(0)

#canvas[200:,200:] = (255,0,0)
green = (0,255,0)
red = (0,0,255)
#Draw a green line from (300, 0)  (or the 300th column and 0th row or top right corner) and (0, 300) (0th column and 300th row or lower left corner). Thickness is 3 pixels
#Draw a red line from (0,0) to (300,300) with thickness of 5
#The last parameter: 3 controls the thickness of the line
cv2.line(canvas, (300,0),(0, 300), green, 3)
cv2.line(canvas, (300,300),(0, 0),red,5)

cv2.imshow("canvas",canvas)
cv2.waitKey(0)

#Let us draw a rectangle
#Here we will draw a shallow rectangle
cv2.rectangle(canvas,(10,20),(20,50),red,2)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)

#Let us draw a rectangle with filled color. -1 width means fill the shape
cv2.rectangle(canvas,(200,40),(300,100),green,-1)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)

#Let us draw a circle:
#Reset the canvas first
canvas = np.zeros((300,300,3),dtype="uint8" )

cv2.circle(canvas, (150, 150), 75, (255, 255, 255), -1)

cv2.imshow("canvas with circle", canvas)
cv2.waitKey(0)
