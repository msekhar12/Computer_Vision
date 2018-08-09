#Module:1
#Chapter: 1.9: Thresholding

#Thresholding is one of the most common (and basic) segmentation techniques in computer vision 
#and it allows us to separate the foreground (i.e. the objects that we are interested in) 
#from the background of the image.

#Types of thresholding
#1. Simple thresholding - We will supply the threshold value
#It works extremely well in controlled lighting conditions, where we can ensure high contract between 
#foreground and background

#2. Otsu's thresholding  - The threshold is synamically calculated

#3. Adaptive thresholding 
#Instead of trying to threshold an image globally using a single value, adaptive thresholding breaks the image down into smaller pieces, 
#and thresholds each of these pieces separately and individually. It is useful in situations where lighting conditions cannot be controlled.

#Thresholding is binarization of the image. In general, we seek to convert a grayscale image to a binary image, where the pixels are either 0 or 255.

#Simple thresholding:
#Set pixels whose values less than a threshold value T to 0 and greater than or equal to T to 255.
#Inverse threshold can also be applied. Inv threshold will make pixels greater than T to 0 and less that or equal to T to 255
#cv2.threshold(image, Threshold, value, type). It returns the "Threshold" and "thresholded image" as output. 
#But the "Threshold" value is the same as the one we supplied as input.
#Where image = image on which the threshold has to be applied
#      Threshold = threshold value
#      value = pixel intensity to be used to set the pixels satisfying the threshold
#      type = cv2.THRESH_BINARY, will set the pixels whose value < Threshold to 0.
#             pixels whose value >= Threshold is set to value
#      type = cv2.THRESH_BINARY_INV, will set the pixels whose value >= Threshold to 0.
#             pixels whose value < Threshold is set to 255


# import the necessary packages
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("Image", image)



#apply basic thresholding -- the first parameter is the image
#we want to threshold, the second value is our threshold check
#if a pixel value is greater than our threshold (in this case,
#200), we set it to be BLACK, otherwise it is WHITE.
(T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)
 
#using normal thresholding (rather than inverse thresholding),
#we can change the last argument in the function to make the coins
#black rather than white. (assuming the input image as coins image)
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)
 
#finally, we can visualize only the masked regions in the image
cv2.imshow("Output", cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)


#The major drawback of basic thresholding or simple thresholding is to supply the threshold value manually.
#Supplying the threshold value manually is not easy, if we have different images taken in different lighting conditions. 
#Hence we will use more automatic thresholding techniques like otsu's and adaptive thresholding.


#Otsu's method
#Otsu’s method assumes that our image contains two classes of pixels: the background and the foreground. 
#Furthermore, Otsu’s method makes the assumption that the grayscale histogram of our pixel intensities of our image is bi-modal, 
#which simply means that the histogram is two peaks.
#A histogram is simply a tabulation or a “counter” on the number of times a pixel value appears in the image.
#Otsu’s method is an example of global thresholding — implying that a single value of T is computed for the entire image. 
#In some cases, having a single value of T for an entire image is perfectly acceptable — but in other cases, this can lead to sub-par results

#There are 2 ways to apply Otsu's thresholding.
#The first method is using mahatos package of python.
import mahotas

#Get the optimal threshold
T = mahotas.thresholding.otsu(blurred)
print("Mahatos determined threshold: {}".format(T))
#Apply the optimal threshold obtained:
T,otsu_thresholded = cv2.threshold(blurred, T, 255,cv2.THRESH_BINARY)

cv2.imshow("otsu_thresholded", otsu_thresholded)
cv2.waitKey(0)

#Inverse otsu threshold:
#Apply the optimal threshold obtained:
T,otsu_thresholded = cv2.threshold(blurred, T, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("otsu_thresholded_inv", otsu_thresholded)
cv2.waitKey(0)



#Another method to apply otsu's threshold using cv2
(T, thresholded) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("thresholded_otsu", thresholded)
cv2.waitKey(0)

    
(T, thresholded_inv) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

cv2.imshow("thresholded_otsu_inv", thresholded_inv)
cv2.waitKey(0)

print("cv2 determined threshold: {}".format(T))

#Otsu thresholding will help us to automatically determine the optimal threshold to separate the foreground objects from background.
#For simple images with controlled lighting conditions, this usually isn’t a problem. 
#But for situations when the lighting is non-uniform across the image, having only a single value of T can seriously hurt our thresholding performance.
#Simply put, having just one value of T may not suffice.

#Adaptive thresholding
#Adaptive thresholding considers small neighbors of pixels and then finds an optimal threshold value T for each neighbor. 
#This method allows us to handle cases where there may be dramatic ranges of pixel intensities and the optimal value of T may change for different parts of the image.
#In adaptive thresholding, sometimes called local thresholding, our goal is to statistically examine the pixel intensity values in the neighborhood of a given pixel p.
#However, choosing the size of the pixel neighborhood for local thresholding is absolutely crucial.
#Which statistic do we use to compute the threshold value T for each region?
#It is common practice to use either the arithmetic mean or the Gaussian mean of the pixel intensities in each region (other methods do exist, but the arithmetic mean 
#and the Gaussian mean are by far the most popular).
#In the arithmetic mean, each pixel in the neighborhood contributes equally to computing T. 
#And in the Gaussian mean, pixel values farther away from the (x, y)-coordinate center of the region contribute less to the overall calculation of T.




###NOTES on threshold using pyimagesearch book:
#Thresholding
#Thresholding is the binarization of an image. In general,
#we seek to convert a grayscale image to a binary image,
#where the pixels are either 0 or 255.

#SIMPLE THRESHOLDING:
#Applying simple thresholding methods requires human intervention.
#We must specify a threshold value T. All pixel
#intensities below T are set to 0. And all pixel intensities
#greater than T are set to 255.
 
#We can also apply the inverse of this binarization by setting
#all pixels below T to 255 and all pixel intensities greater
#than T to 0
 
 
#ADAPTIVE THRESHOLDING:
#One of the downsides of using simple thresholding methods
#is that we need to manually supply our threshold value
#T. Not only does finding a good value of T require a lot of
#manual experiments and parameter tunings, it’s not very
#helpful if the image exhibits a lot of range in pixel intensities.
 
#Simply put, having just one value of T might not suffice.
#In order to overcome this problem, we can use adaptive
#thresholding, which considers small neighbors of pixels
#and then finds an optimal threshold value T for each neighbor.
#This method allows us to handle cases where there
#may be dramatic ranges of pixel intensities and the optimal
#value of T may change for different parts of the image.
 
#OTSU and RIDDLER-CALVARD Thresholding
#Otsu’s method assumes there are two peaks in the grayscale
#histogram of the image. It then tries to find an optimal
#value to separate these two peaks – thus our value of T.
#Even though opencv has support for OTSU's method,
#I (the author/Adrian) prefer to use mahotas method, since it is more pythonic.
#T = mahotas.thresholding.otsu(blurred_image)
#Where T will be the calculated optimal threshold
 
 
import cv2
import numpy as np
import argparse
import mahotas
 
ap = argparse.ArgumentParser()
 
ap.add_argument("-i","--image",required=True,help="Path to image")
 
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])
 
#Convert the image to gray scale, as thresholding is applied to gray scaled images
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#Apply gaussian blurring.
#Applying gaussian blurring will remove some of the
#high frequency edges in the image we are not concerned with
 
blurred_image = cv2.GaussianBlur(image,(15,15),0)
 
#We can apply thresholding using cv2.threshold(image, T, S, type)
#image: Gray scaled image on which the threshold needs to be applied
#T: Threshold value
#S: Value to be set. Any pixel greater than T, will be set to S
#type: The thresholding type
#        cv2.THRESH_BINARY will set the pixel value to S if the pixel is greater than T
#        cv2.THRESH_BINARY_INV will set the pixel value to S if the pixel is less than T
#        cv2.THRESH_TRUNC  leaves the pixel intensities as they are if the source pixel is not greater than the supplied threshold(T)
#        cv2.THRESH_TOZERO sets the source pixel to zero if the source pixel is not greater than the supplied threshold (T)
#        cv2.THRESH_TOZERO_INV sets the source pixel to zero if the source pixel is greater than the supplied threshold (T)
 
#cv2.threshold(image, T, S, method) function will return (Threshold, thresholded_image). The
#"Threshold" is the value of T we used in the function and the "thresholded_image" is the
#converted image.
 
(T, t_image) = cv2.threshold(blurred_image, 240, 255, cv2.THRESH_BINARY)
(T, t_image_INV) = cv2.threshold(blurred_image, 230, 230, cv2.THRESH_BINARY_INV)
 
 
#LET US USE THE INVERSE THRESHOLD AS THE MASK, AND APPLY ON THE ORIGINAL
masked_image = cv2.bitwise_and(image, image, mask=t_image_INV)
 
cv2.imshow("original",image)
cv2.imshow("thresholded",t_image)
cv2.imshow("thresholded_INV",t_image_INV)
cv2.imshow("masked",masked_image)
 
cv2.waitKey(0)
 
#Let us work on adaptive thresholding
#cv2.adaptiveThreshold(image, max_value, method_1, method_2, window, C)
#image: Gray (and blurred image)
#max_value: to be substituted if the pixel exceeds the calculated threshold
#NOTE that we do not supply any threshold here. It is calculated on the fly based on the
#area of enclosure (window)
#method_1: cv2.ADAPTIVE_THRESH_MEAN_C (mean of all pixels within window) will be the threshold (or)
#          cv2.ADAPTIVE_THRESH_GAUSSIAN_C (threshold calculated using the weighted mean of pixels within window)
#method_2: cv2.THRESH_BINARY_INV or cv2.THRESH_BINARY
#window: squared window to be considered as neighborhood
#C: This value is an integer that is subtracted from the mean, allowing us to fine-tune our thresholding
#THE FUNCTION RETURNS CHANGED IMAGE
 
#In general, choosing between mean adaptive thresholding
#and Gaussian adaptive thresholding requires a few experiments
#on your end. The most important parameters
#to vary are the neighborhood size and C, the value you
#subtract from the mean. By experimenting with this value,
#you will be able to dramatically change the results of your
#thresholding.
 
a_threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
19, 6)
 
a_threshold_image_INV = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
19, 6)
 
cv2.imshow("adaptive threshold applied",a_threshold_image)
cv2.imshow("adaptive threshold applied inv",a_threshold_image_INV)
cv2.imwrite("adaptive_threshold_applied.png", a_threshold_image)
cv2.imwrite("adaptive_threshold_applied_INV.png", a_threshold_image_INV)
cv2.waitKey(0)
 
#OTSU
T = mahotas.thresholding.otsu(blurred_image)
 
#Apply the threshold:
threshold = image.copy()
 
 
threshold[threshold > T] = 255
threshold[threshold <= T] = 0
cv2.imwrite("RED_OTSU.png", threshold)
#The following 3 statements, will change the 255 to 0 and 0 to 255.
#Uncomment if needed
threshold[threshold == 255] = 1
threshold[threshold == 0] = 255
threshold[threshold == 1] = 0
print("OTSU's optimal threshold: {}".format(T))
cv2.imwrite("RED_OTSU_INV.png", threshold)
 
 
cv2.imshow("OTSU",threshold)
cv2.waitKey(0)
 
#Riddler-Calvard
T = mahotas.thresholding.rc(image)
#Apply the threshold:
threshold = image.copy()
 
 
threshold[threshold > T] = 255
threshold[threshold <= T] = 0
 
print("RIDDLER-CALVARD optimal threshold: {}".format(T))
cv2.imshow("RIDDLER-CALVARD",threshold)
cv2.imwrite("RED_RIDDLER.png", threshold)
 
cv2.waitKey(0)




#Q. Download the following image: http://pyimg.co/zf3po
#Suppose we wanted to use the cv2.threshold with the cv2.THRESH_BINARY flag. What is an appropriate value to segment the noisy background from the white circle foreground?
#A. 50
#B. 15
#C. 95
#D. 10 

image = cv2.imread("./images/threshold_example.png")
for i in [50, 15, 95, 10]:
    T, new_image = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
    cv2.imshow("new_image_"+str(i), new_image) 
    cv2.waitKey(0)