#Module:1
#Chapter: 1.7 Smoothing and Blurring

#Blurring happens when each pixel in the image mixes with its surrounding pixel intensities.
#This “mixture” of pixels in a neighborhood becomes our blurred pixel.

#Smoothing or Blurring is the same...they are not different concepts
#
#Blurring/Smoothing
#In photography, we try to avoid blurring.
#But in CV, it is needed especially to detect the edges in an image.
#The main goal of blurring methods is to reduce noise and details in image.
 
#There are many ways to introduce blurring.
#AVERAGING Blurring
#GAUSSIAN Blurring
#MEDIAN Blurring
#BILATERAL Blurring
 
 
#AVERAGING Blurring:
#As the name suggests, we are going to define a MXN (M and N must be odd integers)
#sliding window on top of our image
#This window is going to slide from left-to-right
#and from top-to-bottom. The pixel at the center of this matrix
#(we have to use an odd number, otherwise there would
#not be a true "center") is then set to be the average of all
#other pixels surrounding it.
#We call this sliding window a "convolution kernel" or
#just a "kernel".
#Kernels can be an arbitrary size of M x N pixels, provided that both M and N are odd integers.
#Note: Most kernels you’ll typically see are actually square N x N matrices.
 
#IMPORTANT: As the size of the kernel increases, the more blurred our image will become.
#Again, as the size of your kernel increases, your image will become progressively more blurred.
#This could easily lead to a point where you lose the edges of important structural objects in the image.
#Choosing the right amount of smoothing is critical when developing your own computer vision applications.
 
#use cv2.blur(image,(Kx,Ky)) to apply average blurring
#where image = the image on which we are applying the blurring
#      Kx = Number of columns of the kernel matrix
#      Ky = Number of rows of the kernel matrix
 
#A 3X3 kernel used for Average blurring will look like the following matrix:
#  (1/9)[[1,1,1],
#        [1,1,1],
#        [1,1,1]]
 
#GAUSSIAN Blurring:
#Gaussian blurring is similar to average blurring, but instead of
#using a simple mean, we will use a weighted mean,
#where neighborhood pixels that are closer to the central
#pixel contribute more 'weight' to the average and pixels far from the central pixel
#contribute less to the average.
#The end result is that our image is less blurred, but more naturally blurred,
#than using the average method discussed in the previous section.
#Furthermore, based on this weighting we’ll be able to preserve more of the edges in our image as compared to average smoothing.
 
#Just like an average blurring, Gaussian smoothing also uses a kernel of M X N, where both M and N are odd integers.
 
#Std. Normal dist = G(x) = 1/(sqrt(2*pi*) sigma) * e^(-x^2/(2*sigma^2))
#where sigma = std. deviation of x
 
#But in kernel for each point we have 2 variables (x,y) or location.
#Std. Normal dist = G(x) = 1/(sqrt(2*pi*) sigma) * e^(-(x^2+y^2)/(2*sigma^2))
#For more details search gaussian blurring (It uses product of two gaussian distributions)
 
#The image will be less blurred, if Gaussian blur is used (when compared to Average blur)
 
#In cv2 we will use cv2.GaussianBlur(image, (kx, ky), 0)
#where image = input image on which the blur has to be applied
#      (kx, ky) = kernel matrix's dimensions, and must be positive odd numbers
#      0 = To automatically calculate the sigma. But sigma can also be manually supplied.
#      In fact you can supply the sigma on x-direction and sigma on y-direction.
#      See https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur for more details
#But most of the time we want the sigma to be computed automatically. So we will supply 0 as the third parameter.
 
#In general, I (Adrian) tend to recommend starting with a simple Gaussian blur and tuning your
#parameters as needed. While the Gaussian blur is slightly slower than a simple average blur (and only by a tiny fraction),
#a Gaussian blur tends to give much nicer results, especially when applied to natural images.
 
#MEDIAN Blurring:
#Same as AVERAGE blurring, but instead of mean we will use median of all pixels withing the kXk kernel.
#This is very useful to remove salt and pepper kind of noise from the image.
#Unlike gaussian blurring and average blurring, the kernel MUST be a square matrix for Median blurring
#We use cv2.medianBlur(image, k) function to apply median blur
#where image = input image
#      k = kernel matrix dimension, which must be an odd number.
 
#The median blur is by no means a “natural blur” like Gaussian smoothing.
#However, for damaged images or photos captured under highly sub-optimal conditions, a median blur can really help as
#a pre-processing step prior to passing the image along to other methods, such as thresholding and edge detection.
 
 
#BILATERAL Blurring:
#In all the above blurring methods, we tend to lose edges, although we are able to reduce the noise and details
#In order to reduce noise while still maintaining edges, we
#can use bilateral blurring. Bilateral blurring accomplishes
#this by introducing two Gaussian distributions.
 
#The first Gaussian function only considers spatial neighbors,
#that is, pixels that appear close together in the (x, y)
#coordinate space of the image. The second Gaussian then
#models the pixel intensity of the neighborhood, ensuring
#that only pixels with similar intensity are included in the
#actual computation of the blur.
 
#Overall, this method is able to preserve edges of an image,
#while still reducing noise. The largest downside to this
#method is that it is considerably slower than its averaging,
#Gaussian, and median blurring counterparts.
 
#The function cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace) will help to perform bilateral blurring
#where image = input image
#      diameter = diameter of our pixel neighborhood — the larger this diameter is,
#                 the more pixels will be included in the blurring computation.
#                 Think of this parameter as a square kernel size.
#      sigmaColor = color standard deviation. A larger value for sigmaColor means that more colors in the neighborhood
#                   will be considered when computing the blur. If we let sigmaColor get too large in respect to the diameter,
#                   then we essentially have broken the assumption of bilateral filtering — that only pixels of similar color
#                   should contribute significantly to the blur.
#      sigmaSpace = space std deviation. A larger value of sigmaSpace means that pixels farther out from the central pixel
#                   diameter will influence the blurring calculation.
 
#Summary:
#1. The simple average method is fast, but may not preserve edges in images.
#2. Applying a Gaussian blur is better at preserving edges, but is slightly slower than the average method.
#3. A median filter is primarily used to reduce salt-and-pepper style noise as the median statistic is much more
#   robust and less sensitive to outliers than other statistical methods such as the mean.
#4. Finally, the bilateral filter preserves edges, but is substantially slower than the other methods.
#   Bilateral filtering also boasts the most parameters to tune which can become a nuisance to tune correctly.
#5. In general, Adrain recommends starting with a simple Gaussian blur to obtain a baseline and then going from there.
 
 
import cv2
import numpy as np
import argparse
 
ap = argparse.ArgumentParser()
 
ap.add_argument("-i","--image",required=True, help="Path to image")
 
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])
 
#To perform average blurring, we will use cv2.blur(image, (k,k))
#where k is an odd number signifying the sliding window dimensions
#But the kernel can be any 2D matrix as long as the dimensions are odd.
#example: (3,7) or (9,5) etc.
 
a_blurred = np.hstack([cv2.blur(image,(3,3)), \
                    cv2.blur(image,(5,5)), \
                    cv2.blur(image,(7,7))])
                   
#We can observe that as the size of the kernel (window) increases                     ,
#the image gets more blurred.
 
#The np.hstack() will horizontally combine the np matrices.
#This way we can join the images together as one image horizontally, and display
 
#In the following statement we are applying gaussian blur.
#The first parm is image, the second parm is a window,
#and the third parm of 0 signifies the std dev to be used
#to sample the pixels horizontally. But if it is 0, then we are asking cv2 to compute and
#use the std dev as per the kernel size
g_blurred = np.hstack([cv2.GaussianBlur(image,(3,3),0), \
                    cv2.GaussianBlur(image,(5,5),0), \
                    cv2.GaussianBlur(image,(7,7),0)])
 
 
#Applying median blur
m_blurred = np.hstack([cv2.medianBlur(image,3),
                       cv2.medianBlur(image,5),
                       cv2.medianBlur(image,7)])                   
 
#The median blur images show that we are no longer creating a motion blur
#Instead we are removing both noise and detail.
 
#Applying bilateral blur
#The first parameter we supply
#is the image we want to blur. Then, we need to define the
#diameter of our pixel neighborhood. The third argument
#is our color sigma (or std. dev). A larger value for color sigma means that more
#colors in the neighborhood will be considered when computing
#the blur. Finally, we need to supply the space sigma. A
#larger value of space sigma means that pixels farther out from
#the central pixel will influence the blurring calculation, provided
#that their colors are similar enough.
#As the size of our parameters increases,
#our image has noise removed, yet the edges still remain
 
bi_blurred = np.hstack([cv2.bilateralFilter(image, 5, 21, 21),
                        cv2.bilateralFilter(image, 7, 31, 31),
                        cv2.bilateralFilter(image, 9, 41, 41)])
                       
                       
cv2.imshow("original", image)
cv2.imshow("avg blurred", a_blurred)
cv2.imshow("gaussian blurred", g_blurred)
cv2.imshow("median blurred", m_blurred)
cv2.imshow("bilateral blurred", bi_blurred)
cv2.waitKey(0)
 