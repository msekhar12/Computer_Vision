#Module:1
#Chapter: 1.4 Basic image processing
#1.4.4: Flipping

#OpenCV provides methods to flip an image across its x or y axis.
#Flipping is a very important technique to generate artificial data
#For instance if you are working on a face detection algorithm, an if you have
#only a limited number of images containing faces, then you can flip
#your image across y-axis (this is called horizontal flip)
#to generate more images with faces (since a face is still a face,
#no matter if it is mirrored or not) and use these mirrored versions as additional training data.

#To flip the image horizontally use:
#As per my understanding, horizontally means, across the y-axis
#cv2.flip(image, 1)

#To flip the image vertically use:
#As per my understanding, vertically means, across the x-axis
#cv2.flip(image, 0)

#To flip horizontally and vertically use:
#As per my understanding the order of application does not matter.
#You will get the same image if you apply horizontal and then vertical flip (or)
#vertical and then horizontal flip
#cv2.flip(image, -1)

#Example:
#Imagine a matrix:
#A=[[1,2,3],
#    [4,5,6],
#    [7,8,9]]

#Then a horizontal flip will be:
#[[3,2,1],
#[6,5,4],
#[9,8,7]]

#A vertical flip will be:
#[[7,8,9],
#[4,5,6],
#[1,2,3]]

#A horizontal and vertical flip will be:
#[[9,8,7]
#[6,5,4],
#[3,2,1],
#]

#A vertical and horizontal flip will be:
#[[9,8,7],
#[6,5,4],
#[3,2,1]]

#OBSERVE that "A horizontal and vertical flip" is same as "A vertical and horizontal flip"

# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# flip the image horizontally
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)

# flip the image vertically
flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)

# flip the image along both axes
flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)
cv2.waitKey(0)

#Questions:
#Download the source code and images associated with this lesson. Then, use the florida_trip.png to answer the following question.

#Use OpenCV to flip the image horizontally — what is the value of the pixel located at x=259, y=235?
# flip the image horizontally
flipped = cv2.flip(image, 1)
print(flipped[235,259])
#[189 192 183]

#Use the original image from the previous question and flip it horizontally,
#followed by a 45 degree counter-clockwise rotation, and lastly a vertical flip.
#What is (approximately) the pixel value located at x=441, y=189?


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
flipped = cv2.flip(image,1)
rotated = rotate(flipped, 45)
flipped_v = cv2.flip(rotated,0)
print(flipped_v[189, 441])
#[24 19 18]

cv2.imshow("flipped_v", flipped_v)
cv2.waitKey(0)