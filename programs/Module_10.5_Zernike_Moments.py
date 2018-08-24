#10.5: Zernike Moments
 
#Similar to Hu Moments, we can use Zernike Moments to characterize and quantify the shape of an object in an image.
#Also similar to Hu Moments, the shape in an image we wish to describe can be either the outline (i.e. “boundary”) of the shape or
#a mask (i.e. “filled in boundary”) of the shape we want to describe.
#In most real world applications, it’s common to use the shape mask since it’s less susceptible to noise.
#In this program we used mask to eliminate a part of another object from the ROI for the scissors object.
 
#Unlike Hu Moments, Zernike Moments are more powerful and generally more accurate image descriptors with very little additional computational cost.
#The reason Zernike moments tend to be more powerful is because they are orthogonal — this means that there is little-to-no redundancy of information between the moments.
 
#For a detailed description on Zernike moments see Adrain's notes (pyimagesearchgurus)
#Summary:
#Polar coordinates:
#In general we represent z dimensional coordinate system as a cartesian space composed of (x,y) coordinates
#In polar coordinates, we will have angle and radius to represent coordinates in the polar plane.
#To convert polar coordinates to cartesian coordinates:
#x = r cos(theta)
#y = r sin(theta)
#where r = radiua and theta = angle
 
#Zernike polynomials are orthogonal over a disk with radius r (specified in polar coordinates), thus making them applicable to computer vision and shape description.
#To compute Zernike moments for a given shape, we must define a circular region with radius R surrounding the shape we want to describe.
 
# The radius r should technically be set properly to include the entire region of the shape, but in practice this constraint is normally relaxed to ensure images are described in a consistent manner.
 
#To compute Zernike Moments we specify two parameters: the "radius"  of the disc and the "degree"  of the polynomial.
#The radius  is thus the region of which the polynomials are defined.
 
#Pixels that fall outside the disc are ignored and not included in the computation
 
#Zernike Moments up to degree d are computed and utilized as the feature vector.
#The size of the returned feature vector is directly controlled by the degree of the polynomial.
#The larger the degree, the larger the feature vector.
 
#Zernike Moments are implemented inside the mahotas Python package.
 
#To compute zernike moments you can use the following statements:
#import  mahotas
#moments = mahotas.features.zernike_moments(image, 21, degree=8)
 
#The rest of this document will have code to detect an object (reference object) in an image with many objects (one of which is a reference object).
#We will use the object ./images/zernike_reference.jpg as our reference object and ./images/zernike_distractor.jpg as the image with many objects
#The reference object is a pokemon catridge.
 
from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2
import imutils
 
 
#Read the distractor object:
distractor = cv2.imread("./images/zernike_distractor.jpg")
cv2.imshow("distractor_original", distractor)
cv2.waitKey(0)
 
#Convert to gray:
distractor_gray = cv2.cvtColor(distractor, cv2.COLOR_BGR2GRAY)
cv2.imshow("distractor_gray", distractor_gray)
cv2.waitKey(0)
 
 
#Apply blurr (this is a very important step...)
blurred = cv2.GaussianBlur(distractor_gray, (13, 13), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
 
 
#APPLY THRESHOLD:
distractor_canvas = np.zeros(blurred.shape,dtype="uint8")
distractor_canvas[blurred > 50] = 255
cv2.imshow("distractor_canvas", distractor_canvas)
cv2.waitKey(0)
 
#Apply dilation (to close small gaps within the objects)
distractor_canvas = cv2.dilate(distractor_canvas, None, iterations=4)
cv2.imshow("distractor_canvas", distractor_canvas)
cv2.waitKey(0)
 
#Erode the object
distractor_canvas = cv2.erode(distractor_canvas, None, iterations=2)
cv2.imshow("distractor_canvas", distractor_canvas)
cv2.waitKey(0)
 
#See how the scissors became one object
 
#To extract zernike moments, we need to extract the circle enclosing each object in the image
#We do not want to just directly extract the circle, as the circular region of an object might have
#part of another image.
 
 
#The following code will demonstrate this:
 
cnts = cv2.findContours(distractor_canvas.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
#The following code will show that the scissors ROI contains parts of other objects also.
for c in cnts:
    # extract the bounding box ROI from the mask
    (x, y, w, h) = cv2.boundingRect(c)
    roi = distractor_canvas[y:y + h, x:x + w]
    cv2.imshow("roi",roi)
    cv2.waitKey(0)
 
#To avoid that, first we will extract the contoured region, and draw it on a blank canvas,
#and then extract the ROI out of that blank canvas. The extracted object is supplied as input to mahotas.features.zernike_moments() function.
#The second parameter will be the enclosing circle radius. that radius will be obtained from the contour obtained from the original thresholded image.
#We also supply degree=8  to the zernike_moments  function, which is the default degree of the polynomial. In most cases you’ll need to tune this value until it obtains adequate results.
 
 
shapeFeatures = []
 
for c in cnts:
    # create an empty mask for the contour and draw it
    mask = np.zeros(distractor_canvas.shape[:2], dtype="uint8")
    #Since the contour has only outside coordinates, and we are filling the
    #contour completely, the gaps inside the objects (like in scissors) will be completely filled
    cv2.drawContours(mask, [c], -1, 255, -1)
    # extract the bounding box ROI from the mask
    (x, y, w, h) = cv2.boundingRect(c)
    roi = mask[y:y + h, x:x + w]
    cv2.imshow("roi",roi)
    cv2.waitKey(0)
    # compute Zernike Moments for the ROI and update the list
    # of shape features
    features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=4)
    shapeFeatures.append(features)   
 
print(shapeFeatures)   
 
#Let us put the above code inside a function, as the same code needs to be applied for reference image also:
 
# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2
import imutils

def describe_shapes(image):
    # initialize the list of shape features
    shapeFeatures = []

    # convert the image to grayscale, blur it, and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

    # perform a series of dilations and erosions to close holes
    # in the shapes
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)

    # detect contours in the edge map
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
    # loop over the contours
    for c in cnts:
        # create an empty mask for the contour and draw it
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # extract the bounding box ROI from the mask
        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y + h, x:x + w]

        # compute Zernike Moments for the ROI and update the list
        # of shape features
        features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
        shapeFeatures.append(features)

    # return a tuple of the contours and shapes
    return (cnts, shapeFeatures)

cv2.destroyAllWindows()
 
# load the reference image containing the object we want to detect,
# then describe the game region
refImage = cv2.imread("./images/zernike_reference.jpg")
cv2.imshow("refimage",refImage)
cv2.waitKey(0)
 
(_, gameFeatures) = describe_shapes(refImage)

# load the shapes image, then describe each of the images in the image
shapesImage = cv2.imread("./images/zernike_distractor.jpg")
(cnts, shapeFeatures) = describe_shapes(shapesImage)

# compute the Euclidean distances between the video game features
# and all other shapes in the second image, then find index of the
# smallest distance
D = dist.cdist(gameFeatures, shapeFeatures)
i = np.argmin(D)
 
#To prove that the above code is working.
 
# loop over the contours in the shapes image
for (j, c) in enumerate(cnts):
                # if the index of the current contour does not equal the index
                # contour of the contour with the smallest distance, then draw
                # it on the output image
                if i != j:
                                box = cv2.minAreaRect(c)
                                box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
                                cv2.drawContours(shapesImage, [box], -1, (0, 0, 255), 2)

# draw the bounding box around the detected shape
box = cv2.minAreaRect(cnts[i])
box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
(x, y, w, h) = cv2.boundingRect(cnts[i])
cv2.putText(shapesImage, "FOUND!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 3)

# show the output images
cv2.imshow("Input Image", refImage)
cv2.imshow("Detected Shapes", shapesImage)
cv2.waitKey(0)
 
#Questions:
#Q. An important parameter to consider when using Zernike Moments is:
# A. The number of objects in an image.
# B. The size of the radius used.
# C. The Zernike Moments shape descriptor has no parameters to tune.
#Choose one correct option
#ANS: B


 
#Q. Use the image: checkmark.jpg
#   Usind radius 200 pixels and polynomial degree 3, compute the zernike moments.
#   No need to convert this to gray scale
 
image = cv2.imread("./images/checkmark.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
Features = mahotas.features.zernike_moments(image, 200, degree=3)
 
cv2.imshow("checkmark.jpg", image)
cv2.waitKey(0)
 
print(Features)
#[3.18e-01, 3.90e-15, 7.69e-01, 7.28e-02, 1.40e-02, 1.53e-02]
 
 
#3. Question
#In the previous question we are actually doing a poor job of computing Zernike Moments since we are not dynamically computing the radius for each shape.
#True