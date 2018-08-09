#Module:1
#Chapter: 1.10.1: Gradients
 
#We use gradients for detecting edges in images, which allows us to find contours and outlines of objects in images.
#The main application of image gradients lies within edge detection.
#As the name suggests, edge detection is the process of finding edges in an image, which reveals structural information regarding the
#objects in an image. Edges could therefore correspond to:
#Boundaries of an object in an image.
#Boundaries of shadowing or lighting conditions in an image.
#Boundaries of “parts” within an object.
 
#Just as image gradients are building blocks for methods like edge detection,
#edge detection is also a building block for developing a complete computer vision application.
 
#So how do we go about finding these edges in an image?
 
#The first step is to compute the gradient of the image.
 
#Image Gradient: Formally, an image gradient is defined as a directional change in image intensity.
#Or put more simply, at each pixel of the input (grayscale) image, a gradient measures the change
#in pixel intensity in a given direction. By estimating the direction or orientation along with the magnitude
#(i.e. how strong the change in direction is), we are able to detect regions of an image that look like edges.
 
#Finding gradient manually.
#Imagine that we have the following pixel p, with surrounding 4 pixels N (North), S(South), E(East) and W(West)
#       N
#    W  P  E
#       S 
#In the image above we examine the 3 X 3 neighborhood surrounding the central pixel.
#Our x values run from left to right, and our y values from top to bottom. In order to compute any changes in direction
#we’ll need the N (North), S(South), E(East) and W(West)t pixels, which are marked in the above figure
 
#If we denote our input image as I, then we define the north, south, east, and west pixels using the following notation:
#N: I(x, y-1)
#S: I(x, y+1)
#E: I(x+1, y)
#W: I(x-1, y)
 
#The vertical change or y-change or Gy = I(x, y+1) - I(x, y-1)
#The horizontal change or x-change or Gx = I(x+1, y) - I(x-1, y)
#Gx, and Gy represent the change in image intensity for the central pixel "p" in both x and y direction.
 
#Now how are we going to use these values?
 
#To answer that, we’ll need to define two new terms — "the gradient magnitude" and "the gradient orientation".


#The gradient magnitude is used to measure how strong the change in image intensity is.
#The gradient magnitude is a real-valued number that quantifies the “strength” of the change in intensity.
 
#While the gradient orientation is used to determine in which direction the change in intensity is pointing.
#As the name suggests, the gradient orientation will give us an angle that we can use to quantify the direction of the change.
 
#The gradient magnitude is nothing but sqrt(gx^2 + gy^2)
#Based on the pythogarus theorem
 
#The gradient orientation is theta (taninverse or arctan of slope)
#tan(theta) = slope = gy/gx
#theta = tan^(-1)(gy/gx)
 
#To account for various quadrants, we can use arctan2(Gy,Gx) function. This gives the theta in radians.
#So to get the theta in degrees, we need to multiply the radians with 180/pi
#Hence, theta(in degrees) = arctan2(Gy,Gx) * 180/pi
 
#Example:
#Assume that we have the following pixel values surrounding a pixel p
#  255 255 255
#   0   p   0
#   0   0   0
#
#Gx = 0
#Gy = -255
#Gradient magnitude = sqrt(0 + (-255)^2) = 255
#Gradient orientation = arctan2(-255, 0)*180/pi = -90 degrees
#Hence the change in pizel intensity is more towards north.
 
#Sobel and Scharr kernels
#Now that we have learned how to compute gradients manually, let’s look at how we can approximate
#them using kernels, which will give us a tremendous boost in speed. Just like we used kernels to smooth
#and blur an image, we can also use kernels to compute our gradients.
 
#Sobel kernel
#For sobel, we define the Gx and Gy kernels as follows:
#Gx kernel= [[-1, 0, 1],
#            [-2, 0, 2],
#            [-1, 0, 1]]
#
#Gy kernel = [[-1, -2, -1],
#             [ 0,  0,  0],
#             [ 1,  2,  1]]
#
#If you observe Gx = T(Gy) or Gy = T(Gx)
 
#The image pixels are parsed from left to right and from top to bottom using the kernels,
#and each of the central pixel's gradient magnitude and gradient orientation will be calculated using the above Gx and Gy kernels.
#The Gx kernel elements will be multiplied with the corresponding elements of the image and summed up to obtain Gx.
#Similarly Gy kernel will be used to obtain Gy value.
#Then gradient magnitude is canculated as sqrt(Gx^2 + Gy^2)
#The gradient orientation is found as: arctan2(Gy, Gx) * 180/pi
 
#Scharr kernel
#We could also use the Scharr kernel instead of the Sobel kernel which may give us better approximations to the gradient:
#Gx kernel = [[3,  0, -3 ],
#             [10, 0, -10],
#             [3,  0, -3 ]]
#Gy kernel = [[3,  10,   3 ],
#             [0,  0,    0 ],
#             [-3, -10, -3]]
 
#The exact reasons as to why the Scharr kernel could lead to better approximations are heavily rooted in mathematical
#details and are well outside our discussion of image gradients.
 
#Overall, gradient magnitude and orientation make for excellent features and image descriptors when quantifying and
#abstractly representing an image. But for edge detection, the gradient representation is extremely sensitive to local noise.
#We’ll need to add in a few more steps to create an actual robust edge detector — we’ll be covering these steps in detail in the
#next lesson where we review the Canny edge detector.
 
#Sobel kernels in OpenCV
#To compute Sobel gradients, use the following 2 statements:
# compute gradients along the X and Y axis, respectively
#gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
#gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)
 
#cv2.Sobel() is the function to compute the gradient
#gX = Gradient towards x-direction
#gY = Gradient towards y-direction
#gray = Gray scaled image
#dx=1, dy=0 will compute the gradient on x-direction
#dx=0, dy=1 will compute the gradient on y-direction
 
#For Scharr kernel, use cv2.Scharr()
 
#gX, gY are of floating type. We need to convert them to unsigned 8 bit integers
 
#gX = cv2.convertScaleAbs(gX)
#gY = cv2.convertScaleAbs(gY)
 
# import the necessary packages
import argparse
import cv2
import numpy as np
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and display the original
# image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

#compute gradients along the X and Y axis, respectively
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

# the `gX` and `gY` images are now of the floating point data type,
# so we need to take care to convert them back to an unsigned 8-bit

# integer representation so other OpenCV functions can utilize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)
 
 

#combine the sobel X and Y representations into a single image
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

# show our output images
cv2.imshow("Sobel X", gX)
cv2.imshow("Sobel Y", gY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)
 
#Gradient magnitude and Orientation.
# compute the gradient magnitude and orientation respectively
mag = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
 
#We are using % 180 to make sure that the angle is not more than 180 degrees and less than 0 degrees
#For example if we get -2 degrees as the result of np.arctan2(gY, gX) * (180 / np.pi), then
#np.arctan2(gY, gX) * (180 / np.pi) % 180 = -2%180 will be 178 degrees.
 
#print("oreintation: {}".format(orientation))
 
#We can include only pixels whose orientation is within a specific angle range:
# find all pixels that are within the upper and low angle boundaries
#Use upper as 180 and lower as 170
 
#In the following statement, np.where() will get all the Indexes wherever we have
#greater than 170 as the angle. If condifition is satisfied, then the orientation is returned, else -1 is returned as the orientation value
idxs = np.where(orientation >= 170, orientation, -1)
 
#In the following statement, np.where() will get all the IDs wherever we have
#less than 180 as the angle. If condifition is satisfied, then the orientation is returned, else -1 is returned as the orientation value
idxs = np.where(orientation <= 180, idxs, -1)

#Create a mask
mask = np.zeros(gray.shape, dtype="uint8")
 
#Where ever we have more than -1, set to 255 in the mask
mask[idxs > -1] = 255
 
# show the images
cv2.imshow("Mask", mask)
cv2.waitKey(0)
 
#Image gradients are one of the most important image processing and computer vision building blocks you’ll learn about.
#Behind the scenes, they are used for powerful image descriptor methods such as Histogram of Oriented Gradients and SIFT.
#They are used to construct saliency maps to reveal the most “interesting” regions of an image. And as we’ll see in the next lesson,
#we’ll see how image gradients are the cornerstone of the Canny edge detector for detecting edges in images.

#Questions: 
#For the following image, using the North, South, East, and West neighborhood:
#[[44,  67,   96],
# [231, 184, 224],
# [51,  253,  36]]
 
#Compute Gy:
#A: 253 - 67 = 186
 
#Compute Gx  for the following input region of an image:
#A: 224 - 231 = -7
 
#Based on your values of Gx and Gy from the previous two questions, compute the gradient orientation theta:
#A. arctan2(Gy, Gx) * 180/pi = 92
 
#Apply the Gy Sobel kernel for the image region:
#Gy kernel = [[-1, -2, -1],
#             [ 0,  0,  0],
#             [ 1,  2,  1]]
 
##Hence, we will get: -1*44 -2*67 -1*96 + 1*51 +2*253 +1*36 = 319
 
#Apply the Gx Sobel kernel for the following image region:
#Gx kernel= [[-1, 0, 1],
#            [-2, 0, 2],
#            [-1, 0, 1]]
#
##Hence, we will get: -1*44 +1*96 -2*231 +2*224 -1*51 +1*36 = 23
 
#Based on the Gx and Gy from your Sobel convolutions in Question #4 and Question #5, compute the gradient orientation theta.
#theta = np.arctan2(Gy, Gx) * 180/np.pi = 85