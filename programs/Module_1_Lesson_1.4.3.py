#Module:1
#Chapter: 1.4 Basic image processing
#1.4.3: Resizing

#Scaling, or simply resizing, is the process of increasing or decreasing the size of an image in terms of width and height.
#aspect ratio = width/height
#When resizing an image, it’s important to keep in mind the aspect ratio.
#The aspect ratio should remain the same for the original and re-sized image.
#If the aspect ratio is different, then the resized image will look distorted or compressed.
#But there will be cases when we need or can ignore the aspect ratio (more about this in future)

#In general, you’ll want to preserve the aspect ratio of your images when resizing — especially if these images are to be
#presented as output to the user.
#Exceptions most certainly do apply, though, and as we explore machine learning techniques we’ll find that our internal algorithms
#often ignore the aspect ratio of an image; but more on that when we get to machine learning.

#Interpolation is “the method of constructing new data points within the range of discrete set of known points.”
#In general, it’s far more beneficial (and visually appealing) to decrease the size of the image.
#This is because the interpolation function simply has to remove pixels from an image.
#On the other hand, if we were to increase the size of the image the interpolation function
#would have to “fill in the gaps” between pixels that previously did not exist.

#You’ll normally be decreasing the size of an image rather than increasing (exceptions do apply, of course).
#By decreasing the size of the image we have less pixels to process (not to mention less “noise” to deal with),
#which leads to faster and more accurate image processing algorithms.
#The goal of an interpolation function is to examine neighborhoods of pixels and use these neighborhoods
#optically increase or decrease the size of image without introducing distortions (or at least as few distortions as possible).

#We will use cv2.resize(image, dim, interpolation) function to perform resize.
#The dim vlue should be a tuple which has the width and height of resized image.
#We need to manually calculate the new image's width and height so that the aspect
#ratio is maintained

#Interpolation methods:
#Nearest neighbor interpolation: 
#This method is the simplest approach to interpolation. Instead of calculating weighted averages of neighboring pixels or
#applying complicated rules, this method simply finds the “nearest” neighboring pixel and assumes the intensity value.
#While this method is fast and simple, the quality of the resized image tends to be quite poor and can lead to “blocky” artifacts.
#Use interpolation=cv2.INTER_NEAREST to apply this interpolation

#Bilinear interpolation:
#This is the interpolation method opencv uses by default.
#As per my understanding, this method finds a linear function of the form y = mx+c using the neighboring pixels for a given pixel
#Once found, the function will be used to perform the interpolation
#Use interpolation=cv2.INTER_LINEAR to apply this interpolation

#Inter Area interpolation:
#As per Adrian's notes, not much documentation is available about this. However, he mentions that it could be synonymous to
#Nearest neighbor interpolation
#Use interpolation=cv2.INTER_AREA to apply this interpolation

#Inter cubic:
#Similar to linear interpolation, but Cubic splines are used to fit the model to calculate the interpolation.
#Given the computation needed, this method might be slow.
#If you use: interpolation=cv2.INTER_CUBIC, then 4X4 neigbbour pixels will be used to interploate.
#If you use: interpolation=cv2.INTER_LANCZOS4, then 8X8 neigbbour pixels will be used to interploate.
#In general the cv2.INTER_LANCZOS4 method is rarely used in practice.

#In general, cv2.INTER_NEAREST  is quite fast, but does not provide the highest quality results.
#So in very resource-constrained environments, consider using nearest neighbor interpolation — otherwise you probably won’t use
#this interpolation method much (especially if you are trying to increase the size of an image).

#When increasing (upsampling) the size of an image, consider using cv2.INTER_LINEAR  and cv2.INTER_CUBIC.
#The cv2.INTER_LINEAR  method tends to be slightly faster than the cv2.INTER_CUBIC  method, but go with whichever one
#gives you the best results for your images.

#When decreasing (downsampling) the size of an image, the OpenCV documentation suggests using
#cv2.INTER_AREA — although as far as I (Adrian) can tell, this method is very similar to nearest neighbor interpolation.
#In either case, decreasing the size of an image (in terms of quality) is always an easier task than increasing the size of an image.
#Finally, as a general rule, cv2.INTER_LINEAR  interpolation method is recommended as the default
#for whenever you’re upsampling or downsampling — it simply provides the highest quality results at a modest computation cost.

#Example for Aspect Ratio:
#Assume that we have an image with width=300 and height=400.
#We want to resize it to width 450.
#Aspect ratio of existing image = 300/400 = 3/4
#What should be the height of image to maintain 3/4 aspect ratio?
#300/400 = 450/h (where h = required height of resized image to main the aspect ratio)
#h = int(450 * 4/3) = 600 (since the height must be an integer, used int() function)
#hence we will use (450, 600) as our new image's dimensions (width, height)
 
#Import packages:
import cv2
import numpy as np
import argparse

#Create a command line argument to accept image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")

#Read the input arguments
args = vars(ap.parse_args())

#Read the input image
image = cv2.imread(args['image'])

#Let us write a helper function to perform resize:
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
                # initialize the dimensions of the image to be resized and
                # grab the image size
                dim = None
                (h, w) = image.shape[:2]
                # if both the width and height are None, then return the
                # original image
                if width is None and height is None:
                                return image
                # check to see if the width is None
                if width is None:
                                # calculate the ratio of the height and construct the
                                # dimensions
                                r = height / float(h)
                                dim = (int(w * r), height)
                # otherwise, the height is None
                else:
                                # calculate the ratio of the width and construct the
                                # dimensions
                                r = width / float(w)
                                dim = (width, int(h * r))
                # resize the image
                resized = cv2.resize(image, dim, interpolation=inter)
                # return the resized image
                return resized


#Print the dimensions of the image:
print("original image dimensions, in numpy perspective: {}".format(image.shape[:2]))
cv2.imshow("original width:{} height:{}".format(image.shape[1],image.shape[0]), image)
cv2.waitKey(0)
resized_inter_area = resize(image, width=800) #by default uses INTER_AREA interpolation
cv2.imshow("resized INTER AREA width:{} height:{}".format(resized_inter_area.shape[1],resized_inter_area.shape[0]), resized_inter_area)
cv2.waitKey(0)

resized_inter_cubic = resize(image, width=800, inter=cv2.INTER_CUBIC)
cv2.imshow("resized inter cubic width:{} height:{}".format(resized_inter_cubic.shape[1],resized_inter_cubic.shape[0]), resized_inter_cubic)
cv2.waitKey(0)