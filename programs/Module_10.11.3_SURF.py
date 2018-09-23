##10.11.3: SURF
##SURF: Speeded Up Robust Features

##It is very similar to SIFT descriptor, but has two main advantages:
##1. SURF is faster to compute
##2. SURF gives only half the size of the descriptor per key point, when compared to SIFT.
##   SIFT gives 128 features per keypoint, while SURF gives only 64 features per keypoint.

##See the course's 10.11.3 module for more details.



# import the necessary packages
from __future__ import print_function
import argparse
import cv2
import imutils
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the input image, convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# handle if we are using OpenCV 2.4
if imutils.is_cv2():
	# initialize the keypoint detector and local invariant descriptor
	detector = cv2.FeatureDetector_create("SURF")
	extractor = cv2.DescriptorExtractor_create("SURF")
 
	# detect keypoints, and then extract local invariant descriptors
	kps = detector.detect(gray)
	(kps, descs) = extractor.compute(gray, kps)
 
# otherwise, we are using OpenCV 3+
else:
	# initialize the keypoint detector and local invariant descriptor
	detector = cv2.xfeatures2d.SURF_create()
 
	# detect keypoints and extract local invariant descriptors
	(kps, descs) = detector.detectAndCompute(gray, None)
 
# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))



##QUESTIONS:
##1. SURF is faster to compute than SIFT.
##   True

##2. The SURF feature vector is twice as large as SIFT.
##   False

##The speedup from SURF (when compared to SIFT) comes from:
##  A. There is no speedup when compared to SIFT.
##  B. Approximating the image gradient as opposed to explicitly computing it.
##  C. Not allowing for variable size keypoint regions.
##Answer: B

##SURF produces an output feature vector that is:
## A. 128-dim; real-valued
## B. 64-dim; real-valued
## C. 32-dim; binary
## D. 256-dim; real-valued
##Answer: B

