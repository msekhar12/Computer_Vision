from __future__ import print_function
##10.11.2: RootSIFT

##RootSIFT is a simple extension of SIFT descriptor.
##Instead of using L2 normalization, we will L1 normalization, and take square root of each element.

##Step 1: Compute SIFT descriptors using your favorite SIFT library (such as OpenCV).
##Step 2: L1-normalize each SIFT vector.
##Step 3: Take the square root of each element in the SIFT vector.

##It’s a simple extension. But this little modification can dramatically improve results. 
##Whether you’re matching keypoints, clustering SIFT descriptors, or quantizing to form a bag-of-visual-words.

#This can be saved to a python file as a package. 
# import the necessary packages
import numpy as np
import cv2
import imutils
 
class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor for OpenCV 2.4
		if imutils.is_cv2():
			self.extractor = cv2.DescriptorExtractor_create("SIFT")
 
		# otherwise, initialize the SIFT feature extractor for OpenCV 3+
		else:
			self.extractor = cv2.xfeatures2d.SIFT_create()
 
	def compute(self, image, kps, eps=1e-7):
		# compute SIFT descriptors for OpenCV 2.4
		if imutils.is_cv2:
			(kps, descs) = self.extractor.compute(image, kps)
 
		# otherwise, computer SIFT descriptors for OpenCV 3+
		else:
			(kps, descs) = self.extractor.detectAndCompute(image, None)
 
		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)
 
		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
 
		# return a tuple of the keypoints and descriptors
		return (kps, descs)
		
		


# import the necessary packages
#from __future__ import print_function
#from pyimagesearch.descriptors import RootSIFT
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
	detector = cv2.FeatureDetector_create("SIFT")
	extractor = RootSIFT()
 
	# detect keypoints
	kps = detector.detect(gray)
 
# handle if we are using OpenCV 3+
else:
	# initialize the keypoint detector and local invariant descriptor
	detector = cv2.xfeatures2d.SIFT_create()
	extractor = RootSIFT()
 
	# detect keypoints
	(kps, _) = detector.detectAndCompute(gray, None)
 
# extract local invariant descriptors
(kps, descs) = extractor.compute(gray, kps)
 
# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))


##Questions
##RootSIFT is an extension to SIFT that does not require modification to the original SIFT implementation.
##  True


##Why is RootSIFT preferred over the original SIFT implementation?
##  A. None of the above.
##  B. RootSIFT is more suitable for real-time applications.
##  C. It allows SIFT feature vectors to be “compared” using the chi-squared distance — but still utilizing the Euclidean distance.
##  D. RootSIFT is faster to compute than SIFT.
##Answer: C