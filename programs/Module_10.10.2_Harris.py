#10.10.2: Harris
#It is one of the most common corner detector (keypoint detector) used in Computer Vision.
#It is fast (but not as fast as FAST keypoint detector), and is more accurate than FAST keypoints detector.
 
 
#Assume that we have the following set of pixels:
#  0  255
# 255  0
 
#In the first row, a black pixel is followed by a white pixel.
#In the second row a white pixel is followed by a black pixel.
 
#The region lying at the intersection of all the 4 pixels is said to be "corner"
#The goal is to identify such corners.
 
#To detect such corners algorithmically, we just need to take the summation of the
#two gradients: sum(Gx * Gx) in X-direction and sum(Gy * Gy) in Y-direction,
#and consider the intersection of the 4 pixels as a corner if both
#sum(Gx * Gx) and sum(Gy * Gy) are sufficiently large values.
 
#This method works because the region enclosed inside the intersection will have a high number of both
#horizontal and vertical gradients — and when that happens, we know we have found a corner!
 
#To extend this method to arbitrary corners, we first need to
#(1) compute the gradient magnitude representation of an image, and
#(2) then use these gradient magnitude representations to construct a matrix M:
 
#M = [[sum(Gx * Gx), sum(Gx * Gy)],
#               [sum(Gx * Gy), sum(Gy * Gy)]]
 
#Side note:
#How do we calculate Eigen vector and values?
#Assume that we have a matrix:
# A = [[5,   200],
#      [190,   4]]
 
#Then a vector v is said to be eigen vector if A.v = lambda.v
#Where v = eigen vector and lambda = eigen value. The vector v must be a non-zero vector
#That is, transforming a vector v using a matrix A is same as multiplying vector v with a constant lambda.
 
#For a 2x2 matrix, it is easy to find the eigen vectors and values:
#    A.v = lambda.v
#==> (A-lambda.I).v = 0
#==> Since v is non-zero, |(A-lambda.I)| = 0 (Determinant of (A-lambda.I))
#==> A - lambda.I = [[5,   200],  - [[lambda, 0],
#                    [190,   4]]     [0, lambda]]
 
#I assumed A as a matrix with arbitrary elements. NOTE that for harris detector, these values must be "Gradients"
#and NOT the pixel values. AGAIN, the elements must be Gradients Co-variance matrix.
 
#==>A - lambda.I = [[5-lambda,   200],
#                   [190,   4-lambda]]
#
#==> det(A - lambda.I) = (5-lambda) * (4 - lambda) - 38000 = 0
#==> lambda^2 -9.lambda - 37980 = 0
#==> lambda = 199.435 or lambda = -190.435 (approximately)
#The 2 values obtained: 199.435, -190.435 are called eigen values
#Using 199.435 as lambda, we can obtain one eigen vector.
#Let v = [v1, v2] be the eigen vector for the eigen value 199.435.
#Therefore [[5,   200], [190,   4]] * [v1, v2] = [199.435 v1, 199.435 v2]
#==> 5v1 + 200v2 = 199.435v1
#==> v2 = 0.97v1
#==> Since [v1, v2] cannot be [0,0], we can assume v1 = 1 and hance we obtain v2 = 0.97
#We will obtain the same values for v1, v2, if we use the second row equation.
#[[5,   200], [190,   4]] * [v1, v2] = [199.435 v1, 199.435 v2]
#==> 190v1 + 4v2 = 199.435 v2
#==> 190v1 = 195.435 v2
#==> v2 = 0.97v1
#Hence a eigen vector associated to eigen value of 199.435 is [1, 0.97]
 
#Similarly, the eigen vector associated to -190.435 eigen value is:
#[[5,   200], [190,   4]] * [v1, v2] = [-190.435 v1, -190.435 v2]
#==> 5v1 + 200 v2 = -190.435 v1
#==> v2 = (-195.435/200) v1 = -0.97 v1

#==> If v1 = 1, then v2 = -0.97
#Using the second row equation:
#[[5,   200], [190,   4]] * [v1, v2] = [-190.435 v1, -190.435 v2]
#==> 190 v1 + 4 v2 = -190.435 v2
#==> v2 = -0.97 (approximately)
 
#Hence eigen vector associated with -190.435 eigen value is [1, -0.97]
 
#End of NOTE for eigen vectors and values.
 
#To determine if a pixel is a keypoint (or corner) using Harris method, we need to get the eigen values for the
#matrix M (SEE ABOVE).
 
#Let the eigen values for this matrix be lambda_1 and lambda_2.
#Then we compute a value called R as shown below:
#R = det(M) - k(trace(M))^2
#where:
#det(M) = lambda_1 * lambda_2
#trace(M) = lambda_1 + lambda_2
#k = Harris detector free parameter (see https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html)
 
#The value of R is calculated for every pixel in the image.
#To determine if a pixel is a corner, we will perform the following test:
#If |R| is small, we are examining a flat region (and it is not a corner or keypoint). This happens whenever lambda_1 and lambda_2 are small.
#If R < 0 then either lambda_1 >>> lambda_2 or lambda_2 >>> lambda_1. In this case the region is an edge.
#If R is large, then it is a corner. This happens when lambda_1 ~ lambda_2 and both are large.
 
# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils

def harris(gray, blockSize=2, apetureSize=3, k=0.1, T=0.02):
                # convert our input image to a floating point data type and then
                # compute the Harris corner matrix
    #apetureSize is a parameter related to Sobel().
    #As per the documentation it is the size of the extended Sobel kernel; it must be 1, 3, 5, or 7
                gray = np.float32(gray)
                H = cv2.cornerHarris(gray, blockSize, apetureSize, k)
                # for every (x, y)-coordinate where the Harris value is above the
                # threshold, create a keypoint (the Harris detector returns
                # keypoint size a 3-pixel radius)
                kps = np.argwhere(H > T * H.max())
                kps = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kps]

                # return the Harris keypoints
                return kps

# load the image and convert it to grayscale
image = cv2.imread("./images/Station.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting Harris keypoints in the image with OpenCV 2.4
if imutils.is_cv2():
                detector = cv2.FeatureDetector_create("HARRIS")
                kps = detector.detect(gray)

# otherwise we are detecting Harris keypoints with OpenCV 3+ using the function above
else:
                kps = harris(gray)

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
#Harris is used to detect what in images?
#Corners

#To find interesting/salient regions of an image, the Harris keypoint detector relies on the gradient magnitude representation of the input image.
#Yes

#After applying the eigenvalue decomposition of the matrix M, I notice that \lambda_{1} is much larger than\lambda_{2}. This region must be:
#An edge