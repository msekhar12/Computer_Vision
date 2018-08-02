#Module:1
#Chapter: 1.4 Basic image processing
#1.4.7: Bit wise operations

#We can crop and extract a rectangular protion of an image using numpy matrix slicing methods
#But extracting circular or non-rectangular ROI (Region of Interest) is difficult using numpy slicing techniques
#Hence we use bit wise operations and masking techniques to extract any required shape from the image
#We have the following bitwise operations available:

#AND
#OR
#XOR
#NOT

#Although these bit wise operations seem simple, they are very important to understand the concept of masking.

import cv2
import numpy as np
import argparse

#Create a canvas with only one color channel.
canvas_1 = np.zeros((300,300))
canvas_2 = np.zeros((300,300))

#Draw a square
rectangle = cv2.rectangle(canvas_1,(25, 25),(275,275), 255, -1)
 
#Draw a circle
circle = cv2.circle(canvas_2, (150, 150), 150, 255, -1)

#Bit wise operators:
#AND Operation:
# A bitwise 'AND' is only True when both rectangle and circle have
# a value that is 'ON.' Simply put, the bitwise AND function
# examines every pixel in rectangle and circle. If both pixels
# have a value greater than zero, that pixel is turned 'ON' (i.e
# set to 255 in the output image). If both pixels are not greater
# than zero, then the output pixel is left 'OFF' with a value of 0.
#cv2.bitwise_and(image1, image2)

#OR Operation:
# A bitwise 'OR' examines every pixel in rectangle and circle. If
# EITHER pixel in rectangle or circle is greater than zero, then
# the output pixel has a value of 255, otherwise it is 0.
#cv2.bitwise_or(image1, image2)

#exclusive-or/XOR (either A or B but not both)
# The bitwise 'XOR' is identical to the 'OR' function, with one
# exception: both rectangle and circle are not allowed to BOTH
# have values greater than 0.
#cv2.bitwise_xor(image1, image2)

# Finally, the bitwise 'NOT' inverts the values of the pixels. Pixels
# with a value of 255 become 0, and pixels with a value of 0 become
# 255.

#not:
#cv2.bitwise_not(image1)

#applying AND
bitwiseAnd = cv2.bitwise_and(rectangle, circle)

#applying OR
bitwiseOr = cv2.bitwise_or(rectangle, circle)

#applying XOR
bitwiseXOR = cv2.bitwise_xor(rectangle, circle)

#applying not
bitwiseNot = cv2.bitwise_not(circle)

cv2.imshow("rectangle", rectangle)
cv2.imshow("circle", circle)
cv2.imshow("bitwiseAnd", bitwiseAnd)
cv2.imshow("bitwiseOr", bitwiseOr)
cv2.imshow("bitwiseXOR", bitwiseXOR)

 

#Not sure why, but the bitwise NOT is not working correctly.
cv2.imshow("bitwiseNot", bitwiseNot)

cv2.waitKey(0)