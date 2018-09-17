##10.10.7: MSER

##MSER - Maximally Stable Extremal Regions 

##MSER keypoint detector is used to detect “blob-like” regions in an image.

##The MSER detector works by applying a series of thresholds, one for each of the [0, 255] grayscale levels T.

##For each of these levels, a thresholded image is defined by It = I > T, thus constructing a series of black and white thresholded images. 

##Under the hood, MSER is monitoring the changes to each of these thresholded images and is looking for regions of the 
##thresholded images that maintain unchanged shapes over a large set of the possible threshold values.

##Step 1: For each of the thresholded images, perform a connected component analysis on the binary regions.

##Step 2: Compute the area A(i) of each of these connected components.

##Step 3: Monitor the area A(i) of these connected components over multiple threshold values. 
##        If the area remains relatively constant in size, then mark the region as a keypoint.

##The MSER keypoint detector has the benefit of working well on small regions of images that contain distinctive boundaries. 

##The MSER keypoint detector has the benefit of working well on small regions of images that contain distinctive boundaries. 

##When using the MSER detector, be sure to take extra care to ensure the regions of the images you want to detect are 
## (1) small 
## (2) of relatively same pixel intensity, and 
## (3) are surrounding by contrasting pixels.

##The MSER detector tends to be too slow for real-time performance but can exhibit good classification and 
##retrieval performance if the above three scenarios hold.

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
 
# load the image and convert it to grayscale
image = cv2.imread("./images/Station.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect MSER keypoints in the image
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("MSER")
	kps = detector.detect(gray)
 
# otherwise detect MSER keypoints in the image for OpenCV 3+
else:
	detector = cv2.MSER_create()
	kps = detector.detect(gray, None)
 
print("# of keypoints: {}".format(len(kps)))
 
# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)
 
# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)



##Questions:
##1. MSER is used to detect what in images?
##A. Regions with similar pixel intensities.
##B. Corners
##C. Edges
##Answer: A

##2. For a region to be considered a keypoint according to MSER, it must exhibit all of the following properties EXCEPT:
##A. Connected components.
##B. Near uniform pixel intensities.
##C. A high frequency of vertical gradients.
##D. Contrasting background.
##Answer: D

##3. The cornerstone of the MSER algorithms relies on:
##A. Multiple thresholds and a connected-components analysis.
##B. Computing the gradient magnitude representations and finding regions of similar pixel intensity across multiple scales.
##C. Approximating the gradient magnitude through Haar wavelets and applying non-maxima suppression across multiple scales.
##Answer: A


