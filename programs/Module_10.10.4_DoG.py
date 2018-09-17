##10.10.4: DoG (Difference of Gaussian)
##DOG keypoint detector detects blob like regions in the image. 
##The blobs could be corners or edges or combinations of both

##The notion of "scale space" sets this keypoint apart from other keypoints detectors.
##The scale space can help us to identify "repeatable keypoints" irrespective of how close of far the 
##camera is and the angle of the object in the camera. For example, if we are taking a picture of a book, placed on the floor,
##then depending on the distance between book and camera, the size of the book changes in the picture. Also the angle from where 
##the image is captured, also changes the orientation of book in the image.
##The DOG keypoint detector allows us to find "interesting" and "repeatable regions" of an image, even if the scale changes.

#It involves the following process:

##Step-1: The first step in DOG keypoint detection process is to generate the scale space images. 
##        Take the original image and create progressively blurred versions of it. 
##        We then halve the size of the image and repeat. 

##Example:
##Octave_1                Octave_2                         Octave_3                               Octave_4
##--------              ----------------             ----------------------             ------------------------------
##Image_1               Image_1_halved               Image_1_halved_halved              Image_1_halved_halved_halved
##Blurred_1             halved_Blurred_1             halved_halved_Blurred_1            halved_halved_halved__Blurred_1
##Blurred_2             halved_Blurred_2             halved_halved_Blurred_2            halved_halved_halved__Blurred_1
##Blurred_3             halved_Blurred_3             halved_halved_Blurred_3            halved_halved_halved__Blurred_1

##In the above example, octave_1 represents a set of images which were progressively blurred using gaussian blur.
##Then the original image is halved, and applied the gaussian blur progressively again (octave_2)
##The halved image is again halved and progressively blurred for octave_3
##Finally for octave_4 we take the image original image in octave_3, halve and apply the gaussian blur progressively.
##Images that are of the same size (columns) are called octaves.

##Step 2: Difference of Gaussians
##Here, we take two consecutive images in the octave and subtract them from each other. 
##We then move to the next two consecutive images in the octave and repeat the process. 

##For our example, for the first octave, we get:
##Blurred_1 - Image_1   (Let this be DOG_1)
##Blurred_2 - Blurred_1 (Let this be DOG_2)
##Blurred_3 - Blurred_2 (Let this be DOG_3)

##The same difference in images is obtained for other octaves also.

##So if we have n images in an octave, then we will get n-1 images in step 2.

##Step 3: Finding local maxima and minima
##In this step, we will find local maxima and minima in the DoG images.
##
##Each pixel in a DOG (image obtained in the second step) is compared with 8 pixels in its neighbours,
##and also with the 9 pixels in the above DOG and below DOG image.
##For example: Assume that we have the following pixels in DOG_1, DOG_2, DOG_3 (in a 3 X 3 region):
##                    DOG_1
##                         1 2 3 
##                         4 5 5
##                         7 8 9
##
##                    DOG_2
##                         2 3 4
##                         1 P 10
##                         1 4 8
##
##                    DOG_3
##                         2 0 4
##                         1 6 10
##                         1 5 8
##Assume that we are evaluating the pixel P in the above setup. In this scenario, P is compared with all the 8 neigbouring pixels
##and also all the corresponding 9 pixels in the previous layer  (DOG_2) and 9 pixels in subsequent layer (DOG_3), 
##and pixel P is marked as a keypoint if its value is either less than all the 26 neighbouring pixels or more than all its 26 neighbouring 
##NOTE: We get 26 neighbouring pixels, since we have 8 neighbouring pixels in the layer where the pixel being evaluated is present,
##      and 9 neighbouring pixels in the previous layer and 9 in the seusequent layer of the octave.
##

##Finally, we collect all pixels located as maxima and minima across all octaves and mark these as keypoints. 
##Pruning is then performed to remove low contrast keypoints.

##The DoG detector is very good at detecting repeatable keypoints across images, even with substantial changes to viewpoint angle. 
##However, the biggest drawback of DoG is that it’s not very fast and not suitable for real-time applications.
##DoG is still widely used in the computer vision literature and in computer vision applications. 
##And at the very least, it’s practically a requirement for benchmarking new keypoint detector algorithms.

##OpenCV calls DOG as SIFT. But there is a difference between DOG and SIFT (Scale Invariant Feature Transform).
##DOG is a keypoint detector.
##SIFT is feature descriptor which gives a feature vector for each of the keypoing found by DOG.

###I AM NOT ABLE TO TEST THIS CODE AS I WAS GETTING AN ERROR IN THE STATEMENT detector = cv2.xfeatures2d.SIFT_create()

# import the necessary packages

from __future__ import print_function
import numpy as np
import cv2
import imutils
 
# load the image and convert it to grayscale
image = cv2.imread("./images/trex.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect Difference of Gaussian keypoints in the image for OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("SIFT")
	kps = detector.detect(gray)
 
# otherwise we're detecting Difference of Gaussian keypoints for OpenCV 3+
else:
	detector = cv2.xfeatures2d.SIFT_create()
	(kps, _) = detector.detectAndCompute(gray, None)
 
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

##1. DoG is used to detect what in images?
##   A. Blobs.
##   B. Corners.
##   C. Regions with similar pixel intensities.
##   D. Edges.
##   Answer: A

##2. SIFT and DoG have different names, but refer to the same algorithm.
##   False

##3. DoG is able to detect keypoints in images, regardless of scale, by constructing:
##   A. Maxima and minima representations of the gradient approximation.
##   B. Approximating the gradient magnitude of the image.
##   C. Scale space images.
##   Answer: C

##4.The Difference of Gaussians algorithm got its name because:
##  A. It labels regions of an image as keypoints than it determines has the least blurring.
##  B. It approximates a Gaussian blur to an image by subtracting images from consecutive octaves.
##  C. It approximates computing the gradient magnitude by subtracting progressively blurred images from each other.
##Answer: C
  
##5. When finding local maxima and minima, how many checks does the DoG algorithm make to pixels in the current layer, and the layers above and below?
##Answer: 26  