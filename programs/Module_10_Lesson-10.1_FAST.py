#10.10.1 FAST
#FAST is the most simple and intuitive keypoint detector to understand
#It is also one of the fastest keypoint detectors
 
#FAST is used to detect corners in images and is implemented in the OpenCV library and is most
#applicable to real-time applications or resource constrained devices where there is not a lot of
#compute time or power to use more advanced keypoint detectors.
 
#The FAST keypoint detector is used to detect corners in images.
#It is primarily suited for real-time or resource constrained applications where keypoints can quickly be computed.
 
#For a center pixel p to be considered a keypoint, there must be n contiguous pixels that are brighter or darker than
#the central pixel by some threshold t.
#NOTE: We are only testing pixels that fall along the perimeter of the circle!
#In practice, it is common to select a radius of r=3 pixels, which corresponds to a circle of 16 pixels.
#It’s also common to choose n, the number of contiguous pixels, to be either n=9 or n=12.
 
#EXAMPLE:
#In the following pixels, the central pixel "p" has a grqy scale intensity of 32
#For this pixel to be considered a keypoint, it must have n=12 contiguous pixels along the boundary of the
#circle that are all either brighter than p + t or darker than p – t. Let’s assume that t=16 for this example.
#p+t = 32+16 = 48
#p-t = 32-16 = 16
#If there are at least 12 contiguous pixels (lying on the perimeter of a circle with radius = 3 pixels)
#such that:
#         ALL OF THOSE 12 pixels is >= 48 (or)
#         ALL OF THOSE 12 pixels is <= 16
#Then p is considered as a keypoint, else it is not a keypoint.
 
#              191 13 13
#            4           1
#          75             14
#           8       p     11
#          215             1
#           103          15
#              163 141 14  
#
#We can see that none of the 12 contiguous pixels satisfy the intensity constraint. Hence p is NOT a key point in the image.
#Another example:
#In the following scenario also p = 32, radius = 3, n=12 and t = 16.
#              231  27  22
#           212           83
#         136              117
#         123       p      181
#         123              85
#           60          149
#              222 76 126  

#
#If there are at least 12 contiguous pixels (lying on the perimeter of a circle with radius = 3 pixels)
#such that:
#         ALL OF THOSE 12 pixels is >= 48 (or)
#         ALL OF THOSE 12 pixels is <= 16
#We can see that the pixels with intensities 231, 212, 136, 123, 123 60, 222, 76, 126, 149, 85, 181, 117, 83 are all contiguous and greater than 48
#Hence p is a keypoint of the image (and it represents a corner in the image)
 


# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils

# load the image and convert it to grayscale
image = cv2.imread("./images/trex.png")
#image = cv2.imread("./images/grand_central_terminal.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting FAST keypoints in the image for OpenCV 2.4
if imutils.is_cv2():
                detector = cv2.FeatureDetector_create("FAST")
                kps = detector.detect(gray)
# otherwise, we are detecting FAST keypoints for OpenCV 3+
else:
                detector = cv2.FastFeatureDetector_create()
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

#Questions:
#Q. FAST is used to detect what in images?
#Ans: Corners