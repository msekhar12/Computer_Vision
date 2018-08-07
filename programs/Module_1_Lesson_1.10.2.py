#Module:1
#Chapter: 1.10.2: Edge detection
 
#The gradient magnitude and orientation allow us to reveal the structure of objects in an image.
#In later lessons, we’ll see both the gradient orientation and magnitude make for excellent image
#features when quantifying an image.
 
#For the process of edge detection, the gradient magnitude is extremely sensitive to noise.
 
#Assume that we have images and we are interested to draw an outline of the objects inside the image. We can use
#so that we can use contours to extract the objects from the image.
#But if we use gradient representation of the image (like Sobel combined), the resulting image is subject
#to noise, as the gradient representation will also identify the gradients within the object(s), and we are not
#interested in such details.
#Simple image gradients are not going to allow us to (easily) achieve our goal (to extract objects from the image).
#Instead, we’ll have to use the image gradients as building blocks to create a more robust method to detect edges — "the Canny edge detector".
 
#The Canny edge detector
#The Canny edge detector is a multi-step algorithm used to detect a wide range of edges in images.
#The algorithm itself was introduced by John F. Canny in his 1986 paper, A Computational Approach To Edge Detection.
 
#Most image processing projects use Canny edge detector somewhere in the source code.
#An "edge" is defined as discontinuities in pixel intensity,
#or more simply, a sharp difference and change in pixel values.
 
#Types of edges:
#1. Step Edge
#2. Ramp Edge
#3. Ridge Edge
#4. Roof Edge
 
#A step edge forms when there is an abrupt change in pixel intensity from one side of the discontinuity to the other.
#There is a sharp theta step (gradient orientation or theta)
 
#A ramp edge is like a step edge, only the change in pixel intensity is not instantaneous.
#Instead, the change in pixel value occurs a short, but finite distance.
 
#A ridge edge is similar to combining two ramp edges, one bumped right against another. I like to think of ramp edges as driving up and down a large hill or mountain
#First, you slowly ascend the mountain. Then you reach the top where it levels out for a short period. And then you’re riding back down the mountain.
#In the context of edge detection, a ridge edge occurs when image intensity abruptly changes, but then returns to the initial value after a short distance.
 
#Lastly we have the roof edge, which is a type of ridge edge
#Unlike the ridge edge where there is a short, finite plateau at the top of the edge,
#the roof edge has no such plateau. Instead, we slowly ramp up on either side of the edge,
#but the very top is a pinnacle and we simply fall back down the bottom.
 
#Canny algorithm:
#1. Applying Gaussian smoothing to the image to help reduce noise.
#2. Computing the Gx and Gy image gradients using the Sobel kernel.
#3. Applying non-maxima suppression to keep only the local maxima of gradient magnitude pixels that are pointing in the direction of the gradient.
#4. Defining and applying the Tupper and Tlower thresholds for Hysteresis thresholding.
 
#Gaussian blurring will remove any noise in the image
 
#Gx and Gy are gradients, but they are susceptible to noise.
 
#Non-maxima supression: After computing our gradient magnitude representation, the edges themselves are still quite noisy and blurred,
#but in reality there should only be one edge response for a given region, not a whole clump of pixels reporting themselves as edges.
#To remedy this, we can apply edge thinning using non-maxima suppression (procedure given below):
#1. Compare the current pixel to the 3 X 3 neighborhood surrounding it.
#2. Determine in which direction the orientation is pointing:
#    If it’s pointing towards the north or south, then examine the north and south magnitude.
#    If the orientation is pointing towards the east or west, then examine the east and west pixels.
#3. If the center pixel magnitude is greater than both the pixels it is being compared to, then preserve the magnitude. Otherwise, discard it.
 
#Hysteresis thresholding
#Even after applying non-maxima suppression, we may need to remove regions of an image
#that are not technically edges, but still responded as edges applying non-maximum suppression.
#To ignore these regions of an image, we need to define two thresholds: Tupper and Tlower. These are pixel intensities
#For all the pixels remaining after applying the Non-maxima:
#Any pixel value > Tupper is sure an edge pixel
#Any pixel value < Tlower is sure a non-edge pixel
#Any pixel value between Tlower and Tupper needs the following examination:
#If it is attached to an edge, then consider it as an edge, else make it as a non-edge.
 
#Setting these threshold ranges is not always a trivial process.
#If the threshold range is too wide then we’ll get many false edges instead of being about to find just the structure and outline of an object in an image.
#Similarly, if the threshold range is too tight, we won’t find many edges at all and could be at risk of missing the structure/outline of the object entirely!
#The Tupper and Tlower interval can be automatically determined (more on this later in this doc)
 
#The function cv2.Canny(image, Tlower, Tupper) can be used to automatically apply Canny edge detector
 
 
 
# import the necessary packages
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
 
#While Canny edge detection can be applied to an RGB image by detecting edges in each of the separate Red, Green, and Blue channels separately and
#combining the results back together, we "almost always" want to apply edge detection to a single channel, grayscale image. This ensures that there will be less
#noise during the edge detection process.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#While the Canny edge detector does apply blurring prior to edge detection, we’ll also want to (normally)
#apply extra blurring prior to the edge detector to further reduce noise and allow us to find the objects in an image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# show the original and blurred images
cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred)

# compute a "wide", "mid-range", and "tight" threshold for the edges
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

# show the edge maps
cv2.imshow("Wide Edge Map", wide)
cv2.imshow("Mid Edge Map", mid)
cv2.imshow("Tight Edge Map", tight)
cv2.waitKey(0)
 
 
#Depending on your input image you’ll need dramatically different hysteresis threshold values — and tuning these values can be a real pain
 
#Automatically tuning edge detection parameters
#This program is available in imutils package (developed by Adrain at PyImageSearch)
 
def auto_canny(image, sigma=0.33):
                # compute the median of the single channel pixel intensities
                v = np.median(image)
                # apply automatic Canny edge detection using the computed median
                lower = int(max(0, (1.0 - sigma) * v))
                upper = int(min(255, (1.0 + sigma) * v))
                edged = cv2.Canny(image, lower, upper)
                # return the edged image
                return edged

  
#The above function will only include the pixels which fall within +/- 67% (from the median pixel intensity of the image).
#As per Adrian, the default sigma works well for many images, although we should still test the performance for various values of sigma.
 
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = imutils.auto_canny(blurred)

# show the images
cv2.imshow("Original", image)
cv2.imshow("Wide", wide)
cv2.imshow("Tight", tight)
cv2.imshow("Auto", auto)
cv2.waitKey(0)   
 
 
#Questions 
#The gradient magnitude representation of an image is very noisy and makes for a poor edge detector.
#True
 
#The follow are all types of edges except:
# Step edge
# Ramp edge
# Roof edge
# Corner edge
 
#A. Corner edge
 
#Arguably, the most well known edge detector is the Canny edge detector.
#True
 
#All of the following are steps in the Canny edge detector except:
# Computing histogram of edge orientations.
# Non-maxima suppression.
# Gaussian smoothing.
#  Hysteresis thresholding.
 
#A. Computing histogram of edge orientations
 
#A weak edge can still be considered an “edge” (according to the Canny edge detector) if it is connected to a strong edge.
#True