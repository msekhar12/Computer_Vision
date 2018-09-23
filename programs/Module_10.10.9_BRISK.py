##BRISK Keypoint detector

##BRISK: Binary Robust Invariant Scalable Keypoint 

##The BRISK keypoint detector is used to detect corners in images. It is simply a multi-scale version of the FAST keypoint detector.

##It is an extension of FAST keypoint detector (https://github.com/msekhar12/Computer_Vision/blob/master/programs/Module_10_Lesson-10.1_FAST.py)

##The FAST keypoint detector exampines the image at a single scale, meaning that it was unlikely keypoints could be repeated over 
##multiple scales of the image.

##In BRISK, we apply FAST keypoints detector for images at various sizes. We first apply FAST keypoint detector on the image (assume its size is x X y).
##Then we resize the image into x/2 and y/2 dimensions, and again apply FAST keypoint detector
##Then resize the original image to x/4 and y/4 dimensions, and apply FAST keypoint detector.
##Then resize the original image to x/8 and y/8 dimensions, and apply FAST keypoint detector.

##So as you can see, the BRISK keypoint detector is a fairly simple extension to the FAST keypoint detector. 
##However, the major benefit of using BRISK is that scale space invariance is added at only a slightly modest cost 
##(i.e. the computation of the image pyramid and FAST ran on each level of the pyramid). 
##BRISK is still very much suitable for real-time application, so any time you are considering using FAST, 
##also give BRISK a try and see if performance improves.

#import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
 
#load the image and convert it to grayscale
image = cv2.imread("./images/Station.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#detect BRISK keypoints in the image for OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("BRISK")
	kps = detector.detect(gray)
 
#detect BRISK keypoints in the image for OpenCV 3+
else:
	detector = cv2.BRISK_create()
	kps = detector.detect(gray, None)
 
print("# of keypoints: {}".format(len(kps)))
 
#loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)
 
#show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)



##Questions
## 1. BRISK is used to detect what in images?
##    A. Blobs.
##    B. Corners.
##    C. Edges.
##    D. Regions with similar pixel intensities.
##    Answer: B

## 2. BRISK is an extension to the FAST keypoint detector
##    True


## 3. BRISK adds what to the FAST detector:
##    A. Scale invariance
##    B. Illumination invariance
##    C. Viewpoint invariance
##    D. Rotation invariance
## Answer: A

