##10.11.1: SIFT

##Local feature descriptors are broken down into two phases:

##Detecting keypoints (these may correspond to edges or blobs or corners)
##After identifying the set of keypoints in an image, we then need to extract and quantify 
##the region surrounding each keypoint. 

##The feature vector associated with a keypoint is called a feature or local feature since only the 
##local neighborhood surrounding the keypoint is included in the computation of the descriptor.

##For SIFT feature description algorithm, we need to supply a set of keypoints as input to the algorithm.
##Then for each keypoint, we get a feature vector. The feature vector is calculated as follows:

##Foe each keypoint:
##1. Take 16 X 16 srea surrounding the keypoint
##2. Divide the 16 X 16 area into 16 squares (each square is 4 X 4 pixels, such that each square will have 16 pixels).
##   Let us address each of the square as a window
##3. For each window, we get the gradient magnitude and orientation, just like we did for HOG descriptor.
##4. Given the gradient magnitude and orientation, we construct 8-bin histogram for each of the 4X4 pixel windows.
##   The amount added to each bin is dependent on the magnitude of the gradient. However, we are not going to use the 
##   raw magnitude of the gradient. Instead, we are going to utilize Gaussian weighting. 
##   The farther the pixel is from the keypoint center, the less it contributes to the overall histogram.
##5. Finally, we collect all 16 of these 8-bin orientation histograms and concatenate them together, giving us 128 dimensions (or features) for a keypoint.
##6. Once we have collected the concatenated histograms, we end up L2-normalizing the entire feature vector. 

##Again, it’s important to note that unlike global image descriptors such as Local Binary Patterns, 
##Histogram of Oriented Gradients, or Haralick texture (where we have only one feature vector extracted per image), 
##local descriptors return N feature vectors per image, where N is the number of detected keypoints. 
##This implies that given N detected keypoints  in our input image, we’ll obtain N x 128-d feature vectors after applying SIFT.

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
	detector = cv2.FeatureDetector_create("SIFT")
	extractor = cv2.DescriptorExtractor_create("SIFT")
 
	# detect keypoints, and then extract local invariant descriptors
	kps = detector.detect(gray)
	(kps, descs) = extractor.compute(gray, kps)
    
# otherwise, handle if we are using OpenCV 3+
else:
	# initialize the keypoint detector
	detector = cv2.xfeatures2d.SIFT_create()
 
	# detect keypoints and extract local invariant descriptors
	(kps, descs) = detector.detectAndCompute(gray, None)
 
# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))    

##QUESTIONS:
##BOTH SIFT and HOG construct a histogram of gradients where each bin is based on the orientation of the gradient:
##   True

##For each keypoint detected, SIFT extracts:
##  A 16×16 pixel region split into 4×4 cells.


##The amount added to each bin in the gradient histogram is based on the:
## A. Size of the cell.
## B. Magnitude of the gradient.
## C. Size of the keypoint.
## D. None of the above.
##Answer: B

##SIFT produces an output feature vector that is:
## A. 256-dim; real-valued
## B. 32-dim; binary
## C. 64-dim; real-valued
## D. 128-dim; real-valued
##Answer: D



