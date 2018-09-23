##10.10.10: ORB

##ORB: ORB (Oriented FAST and Rotated BRIEF)

##ORB, just like BRISK, is an extension to the FAST keypoint detector. 
##It, too, utilizes an image pyramid to account for multi-scale keypoints; 
##however, (and unlike BRISK), ORB adds rotation invariance as well.

##The first step is to compute the FAST keypoint detector across multiple scales of the image, 
##like in the BRISK keypoint detector. Again, a circular neighborhood of 16 pixels is utilized, 
##where 9 contiguous pixels must be all smaller or larger than the center pixel. 
##If this contiguous case holds, then the center pixel is marked as a keypoint.

##Second step involves applying Harris detector
##All keypoints gathered over all scales of the image pyramid are ranked and sorted according to their Harris keypoint score.
##A maximum of n=500 keypoints are kept, and the rest are discarded.

##Third step adds rotational invariance.
##For each detected keypoint in step:2 we perform the following:
##The region surrounding the center of the keypoint is examined — this process of measuring the orientation of the keypoint is called the "intensity centroid"

##m(p,q) = sum(x**p * y**q) *I(x,y)
##Where x, y are pixel locations, and I(x,y) is intensity of pixel at location (x, y)

##The centroid of the image patch is:
##C = (m(1,0)/m(0,0), m(0,1)/m(0,0))

##Using the intensity centroid, the keypoint can be rotated according to its dominant axis, 
##ensuring that we obtain a canonical representation of the area surrounding the keypoint:

##theta = arctan2(m(0,1), m(1,0))

##Performing this rotation ensures that, if the same keypoint appeared in an image but was rotated by 
##some angle theta, the same keypoint could not only be detected, but also described in the same way.

##In summary, the ORB keypoint detector is built on FAST and shares many commonalities with BRISK, including scale invariance. 
##However, ORB takes an extra step and provides rotation invariance as well.

##ORB is a very fast keypoint detector, and just like FAST and BRISK, it is suitable for real-time applications.

##However, when developing your own applications, there is rarely a clear choice when deciding between 
##FAST, BRISK, and ORB. All detectors have their merits, and since they all have FAST at their core, they produce similar results. 

##My best suggestion (Adrain) is to try each of them and see which detector gives you the best results.

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
 
# load the image and convert it to grayscale
image = cv2.imread("./images/Station.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect ORB keypoints in the image for OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("ORB")
	kps = detector.detect(gray)
 
# detect ORB keypoints in the image for OpenCV 3+
else:
	detector = cv2.ORB_create()
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

##ORB is used to detect what in images?
##  A. Regions with similar pixel intensities.
##  B. Edges.
##  C. Corners.
##  D. Blobs.
##Answer: C

##ORB builds upon FAST and adds which of the two following properties:
##A. Illumination and viewpoint invariance
##B. Scale and rotation invariance
##C. Rotation and illumination invariance
##D. Viewpoint and scale invariance
##Answer: B

