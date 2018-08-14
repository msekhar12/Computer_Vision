#Module:1
#Chapter: 1.6 Morphological operations

#Morphological operations
#Morphological operations are simple transformations applied to binary or grayscale images.
#More specifically, we apply morphological operations to shapes and structures inside of images.
#We can use morphological operations to increase the size of objects in images as well as decrease them.
#We can also utilize morphological operations to close gaps between objects as well as open them.

#Morphological operations "probe" an image with a "structuring element".
#"Probe" means examine.
#The "structuring element" will define the neighborhood to be examined.

#Important morphological operations are given below:
# Erosion
# Dilation
# Opening
# Closing
# Morphological gradient
# Black hat
# Top hat (or "White hat")

#Often times computer vision researchers and developers trying to solve a problem
#immediately dive into advanced computer vision and machine learning techniques.
#It seems that once you learn to wield a hammer, every problem looks like a nail.
#However, there are times where a more "elegant" solution can be found using less advanced techniques.
#And more than likely, you may find that elegant solution in morphological operations.

######################
#Structuring Element #
######################
#Well, you can (conceptually) think of a structuring element as a type of kernel or mask.
#However, instead of applying a convolution (multiplying corresponding elements and summing them), we are only going to
#perform simple tests on the pixels.
#And just like in image kernels, the structuring element slides from left-to-right and
#top-to-bottom for each pixel in the image. Also just like kernels, structuring elements can be of
#arbitrary neighborhood sizes.

#Example
#A 4 pixel neighborhood surrounding:
#          N
#        N C N
#          N
#The 4-neighborhood defines the region surrounding the central pixel as the pixels to the north, south, east, and west.

#An 8 pixel neighborhood surrounding:
#        N N N
#        N C N
#        N N N
#    
#Where C = current pixel, and N = neighbourhood pixel

#The 8-neighborhood extends 4 neighborhood pixel's region to include the corner pixels as well.
#
#This is just an example of two simple structuring elements.
#But we could also make them arbitrary rectangle or circular structures as well.
#It all depends on your particular application.
#
#Understand that a structuring element behaves similar to a kernel or a mask — but instead of convolving the input image
#with our structuring element, we’re instead only going to be applying simple pixel tests.
#
#########
#Erosion#
#########
#Just like water rushing along a river bank erodes the soil,
#an erosion in an image "erodes" the foreground object and makes it smaller.
#Simply put, pixels near the boundary of an object in an image will be discarded, 'eroding' it away.
#Erosion works by defining a structuring element and then sliding this structuring element from
#left-to-right and top-to-bottom across the input image.
#A foreground pixel in the input image will be kept only if ALL pixels inside the
#structuring element are > 0. Otherwise, the pixels are set to 0 (i.e. background).
#That is, if there is at least one black pixel inside the structuring element window, then replace the central pixel 
#(pixel at the center of structuring element) with black pixel.
#So this means erosion assumes the background as black and fore ground as anything else other than black?
#Absolutely. As per my understanding, irrespective of the background or foreground, erosion will always replace
#central pixel inside the kernel with black pixel, if at least one pixel in the kernel is black.
#NOTE that morphological operations are generally applied for binary images.
#But some morphological operations like Tophat, blackhat are generally applied for gray scaled images.
#
#Erosion can be applied to both binary and gray scale images. But what happens if we apply to colored images?
#No problem. We can apply erosion to colored images also, and we will see the same effect: Replace central cell in the kernel
#with black pixel, if all the cells in the kernel are NOT black. So if at least one pixel is black in the kernel, replace
#the kernel's central pixel with black

#Erosion is useful for removing small blobs (non-black colored ones) in an image or disconnecting two connected objects.
#We will use cv2.erode() to apply erosion
#cv2.erode(image.copy(), kernel, iterations=n)
#kernel can be None, in which case a kernel size of 3X3 will be used
#The last argument is the number of iterations  the erosion is going to be performed.
#As the number of iterations increases, we’ll see more and more of the image eaten away.

#Applying erosion on pyimagesearch logo

#Read the image

import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

#Convert to GRAY:
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("original",image)

cv2.imshow("gray",gray)

 
for i in range(0, 3):
    eroded = cv2.erode(gray.copy(),None, iterations=i+1)
    cv2.imshow("eorded",eroded)
    cv2.waitKey(0)
 


###########
#Dialation#
###########
#The opposite of an erosion is a dilation. Just like an erosion will eat away at the foreground pixels, a dilation will grow the foreground pixels.
#Dilations increase the size of foreground object and are especially useful for joining broken parts of an image together.
#Dilations, just as an erosion, also utilize structuring elements — a center pixel p of the structuring element is set to
#white if ANY pixel in the structuring element is > 0.

#cv2.dilate(gray.copy(), None, iterations=n) can be used to apply dialation
#Unlike an erosion where the foreground region is slowly eaten away at, a dilation actually grows our foreground region.
#Dilations are especially useful when joining broken parts of an object
#Irrespective of the background or foreground, dilation will always add white pixels in the kernel if there is one pixel which is not black.

#This is applicable to coloured images also, although we generally apply dilation on binary images.

#REMEMBER:
#Erosion: If there is at least one black pixel inside the kernel window, then the central pixel in the kernel window will be replaced by black pixel
#         In other words all the pixels in the kernel window must have > 0 value to keep the central pixel (in kernel) the same or as currently existing 
#Dilation: If there is at least one non-block pixel in the kernel window then replace the central pixel in the kernel with white pixel.
#         In other words all the pixels in the kernel window must be 0 to keep the central pixel (in the kernel) intact from dilation operation

#Although I used the word "kernel" in this document, always use "structuring element" while refering to morphological operations.
#So wherever I mentioned Kernel, read it as "structuring element". "Kernel" is usually related to convolutional operations.

for i in range(0, 3):
    dilated = cv2.dilate(gray.copy(),None,iterations=i+1)
    cv2.imshow("dilated",dilated)
    cv2.waitKey(0)

#########   
#Opening#
#########
#An opening is an erosion followed by a dilation.
#Performing an opening operation allows us to remove small blobs from an image:
# first an erosion is applied to remove the small blobs (which are non-black blobs), then a dilation is applied to regrow the size of the original object.

#cv2.getStructuringElement(first_parm, second_parm) will create a structuring element for you automatically.
#the first_parm can be cv2.MORPH_RECT (for rectangular kernel or 8 element kernel), cv2.MORPH_CROSS (cross kernel or + sign kernel)
#cv2.MORPH_ELLIPSE (for ellipse shape).
#You can also manually create the structuring element. It is just a numpy array.
#for example to create 5X5 rectangle:
#array([[1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1]], dtype=uint8)
#cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) will return the above numpy matrix

#cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#will show:
#array([[0, 0, 1, 0, 0],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1],
#       [0, 0, 1, 0, 0]], dtype=uint8)

#cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
#will show:
#array([[0, 0, 1, 0, 0],
#       [1, 1, 1, 1, 1],
#       [0, 0, 1, 0, 0]], dtype=uint8)
#In the above example, observe that (5,3) has generated 5 columns and 3 rows for the structuring element (as per cv notation x-axis followed by y-axis)
 
#We will use cv2.morphologyEx(image, morphology_type, kernel) to execute any morphological application.
#image = input image
#morphology_type can be cv2.MORPH_OPEN (for Opening) or cv2.MORPH_CLOSE (for Closing) or cv2.MORPH_GRADIENT
#or cv2.MORPH_TOPHAT or  cv2.MORPH_BLACKHAT
#This function is abstract in a sense — it allows us to pass in whichever morphological operation we want, followed by our kernel/structuring element.
#--------------------------------------------------------
#Remember: "opening" is "erosion followed by dialation"
#--------------------------------------------------------

kernels = [(3,3),(5,5),(7,7)]
for kernelsize in kernels:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelsize)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening",opening)
    cv2.waitKey(0)

#########   
#Closing#
#########
#The exact opposite to an opening would be a closing.
#A "closing" is a "dilation followed by an erosion".
#As the name suggests, a closing is used to close holes
#inside of objects or for connecting components together.

#opening is used to close blobs (which are white in color), while closing is used
#to close small gaps (or black spots) inside a white object.

#To destroy all active windows of cv2.imshow(), use cv2.destroyAllWindows()
cv2.destroyAllWindows()
cv2.imshow("Original", image)

#Performing the closing operation is again accomplished by making a call to cv2.morphologyEx,
#but this time we are going to indicate that our morphological operation is a
#closing by specifying the cv2.MORPH_CLOSE flag.

#-------------------------------------------------------------
#Remember: A "closing" is a "dilation followed by an erosion".
#-------------------------------------------------------------
for kernelsize in kernels:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelsize)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing",closing)
    cv2.waitKey(0)

############################
#Morphological Gradient
############################
#A morphological gradient is the difference between the dilation and erosion.
#It is useful for determining the outline of a particular object of an image:

#First dilation is applied on the image "I"
#Second erosion is applied on the same image "I"
#The difference between first image and second image will give outline of the objects in the original image

for kernelsize in kernels:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelsize)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("gradient",gradient)
    cv2.waitKey(0)

############################
#Top Hat/White Hat
############################
#A top hat (also known as a white hat) morphological operation is the difference between the original input image and the "opening".
#Opening will close small non-black blobs (since erosion will close such small non-black blobs, and dilation almost re-construct the image back to
#original, but without the small non-black blobs. So the difference between original and opening will be more brighter non-black blobs in the original image

#A top hat operation is used to reveal bright regions of an image on dark backgrounds.

#Up until this point we have only applied morphological operations to binary images.
#But we can also apply morphological operations to grayscale images as well.
#In fact, both the top hat/white hat and the black hat operators are more suited for grayscale images rather than binary ones.

#My note, as per my analysis:
#Gray scale images are different from binary. In binary image, we either have 0 or 255 pixel values
#In gray scale and binary there will not be any color channel (so they are 2X2 matrices).
#But gray scale image can have any pixel intensity between 0 and 255

 

#NOTE: when you convert a color to gray scale and save the gray scale image (which is 2D) as jpg or png image, then
#the saved gray scaled image will be saved with three channels, as when we read back that saved gray scaled image, you will see
#3 channels. As per my understanding all the three channels of gray scaled image will be the same (in fact i used this logic
#in one of the projects to test if the image is black and white)

#Imagine a car's license plate on a black coloured car.
#The license plate will be on a white back ground and with black colored letters.
#Create a rectangular kernel with 13 X 5 (13 columns and 5 rows numpy matrix, or 13 on x-axis and 5 on y-axis)

# construct a rectangular kernel and apply a blackhat operation which
# enables us to find dark regions on a light background
# observe that we supplied 13 as the first value (x-axis of the image), and 5 as the second value (y-axis of the image)
#Also note that as our goal is to identify license plate, and usually the width of license plate of car is 3 times its height.
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

cv2.imshow("tophat", tophat)
cv2.waitKey(0)

#When we apply the White Hat on the gray scaled image of the car, then the license plate will be highlighted.
#But the problem is the letters will still remain black on the white background of the plate.
#Moreover, if the car itself is white then it will be even more difficult to use the white hat morphing operator.
#So apply black hat morphing operator
#One thing that we can (almost always) rely on is that the license plate text
#itself being darker than the license plate background. To this end, we can apply a black hat operator cv2.MORPH_BLACKHAT

###############################
#Black Hat operator
###############################
#The black hat operation is the difference between the closing of the
#input image and the input image itself. In fact, the black hat operator
#is simply the opposite of the white hat operator!
 
#We apply the black hat operator to reveal dark regions (something like the license plate text) against light backgrounds
#(i.e. the license plate itself).
 
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
 
cv2.imshow("blackhat", blackhat)
cv2.waitKey(0)
 
 
#Summary (straight from Adrian's notes)
#In this lesson we learned that morphological operations are image processing transformations applied to either grayscale or binary images.
#These operations require a structuring element, which is used to define the neighborhood of pixels the operation is applied to.
#We also reviewed the most important morphological operations that you’ll use inside your own applications:
#erosion
#dilation
#opening
#closing
#morphological gradient
#top hat/white hat
#black hat
 
#Morphological operations are commonly used as pre-processing steps to more powerful computer vision solutions such as OCR,
#Automatic Number Plate Recognition (ANPR), and barcode detection.
 
#While these techniques are simple, they are actually extremely powerful and tend to be highly useful when pre-processing your data.


#Questions:
#1. What is the difference between an erosion and dilation?
#A. An erosion eats away at the foreground object, while a dilation increases the size of the object.
 
#2. What is the difference between a closing and opening operation?
#A. An opening is an erosion followed by a dilation, while a closing is a dilation followed by an erosion.
 
#3. Morphological operations can only be applied to binary images.
#A. False
 
#4. Define a rectangular structuring element with 5 columns and 20 rows.
#A. rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
 
#5. Which kernel shape is NOT listed for cv2.getStructuringElement function?
#A. MORPH_CIRCLE