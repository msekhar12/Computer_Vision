#Module 10.4 Hu Moments
# import the necessary packages
import cv2
import imutils
import mahotas

# load the input image and convert it to grayscale
image = cv2.imread("./images/planes.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Hu Moments feature vector for the entire image and show it
# Observe that we are supplying the cv2.moments() output as input to cv2.HuMoments()
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print("ORIGINAL MOMENTS: {}".format(moments))
cv2.imshow("Image", image)
cv2.waitKey(0)
 
 
 
# The Hu moments computation is made using the complete image.
# But we should use the individual objects Hu moments.
# Hu moments at the image level are not useful.
 
#When we take the entire image into account like this, our shape statistics become completely irrelevant.
#Since Hu Moments are defined relative to their centroid, the centroid becomes the center of all
#three shapes rather than just one of them.
 
#The following code will display the centroid used to compute the above Hu Moments
M = cv2.moments(image)
cx = int(M["m10"]/M["m00"])
cy = int(M["m01"]/M["m00"])
 
cv2.circle(image, (cx, cy), 5,(0, 255, 0), -1)
cv2.imshow("Image", image)
cv2.waitKey(0)
 
 
#To correctly compute Hu Moments for each of the three aircraft silhouettes, we’ll need to find the contours of each airplane,
#extract the ROI surrounding the airplane, and then compute Hu Moments for each of the ROIs individually:
 
#Read the image again:
image = cv2.imread("./images/planes.png")
 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# find the contours of the three planes in the image
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
   
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over each contour
for (i, c) in enumerate(cnts):
    # extract the ROI from the image and compute the Hu Moments feature
    # vector for the ROI
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    #To show centroid of each object (uncomment the following code)
    #M = cv2.moments(roi)
    #cx = int(M["m10"]/M["m00"])
    #cy = int(M["m01"]/M["m00"])
    #cv2.circle(roi, (cx, cy), 5,(0, 255, 0), -1)
    #cv2.imshow("ROI #{}".format(i + 1), roi)
    #cv2.waitKey(0)
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    #show the moments and ROI
    print("MOMENTS FOR PLANE #{}: {}".format(i + 1, moments))
    cv2.imshow("ROI #{}".format(i + 1), roi)
    cv2.waitKey(0)
 
   
#READ the lesson for a outlier image search. Also read my notes in the notebook   
    
#For exercise:
 
image = cv2.imread("./images/shape_explosion.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
cv2.imshow("explosion.png", image)
cv2.waitKey(0)   
 
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
   
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
(x, y, w, h) = cv2.boundingRect(cnts[0])
roi = image[y:y+h,x:x+w]
moments = cv2.HuMoments(cv2.moments(roi)).flatten()
cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255), 3)
 
cv2.imshow("explosion.png", image)
cv2.waitKey(0)   
 
print("explosion image moments:{}".format(moments))
 
image = cv2.imread("./images/more_shapes_example.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
T = mahotas.thresholding.otsu(image)
image[image > T] = 255
image[image <= T] = 0
 
#To get the hu moments of circle only...let us get the contour first, bounding boxes next and finally find the aspect ratio
#ar = width / height
#See https://github.com/msekhar12/Computer_Vision/blob/master/programs/Module_1_Lesson_1.11.3_adv_contour_prop.py
#ar must be approximately 1 for circle, and given that we do not have no shape like square in the image , we can use ar
 
_,cnts,_ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
 
for c in cnts:
    area = cv2.contourArea(c)   
    (x, y, w, h) = cv2.boundingRect(c)
    #extent = area / float(w * h)
    ar = w/h
    if ar >= 0.99 and ar <= 1.01:
       roi = image[y:y+h, x:x+w]
       moments = cv2.HuMoments(cv2.moments(roi)).flatten()
       cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255), 3)
       print(moments)
       print("ar: {}".format(ar))
       cv2.imshow("roi",roi)
       cv2.waitKey(0)
 
 
cv2.imshow("more images.png", image)
cv2.waitKey(0)   
 
#READ the chapter 10.4: Hu Moments completely.
 
 
#Hu Moments Pros and Cons
 
#Pros:
#Very fast to compute.
#Low dimensional.
#Good at describing simple shapes.
#No parameters to tune.
#Invariant to changes in rotation, reflection, and scale.
#Translation invariance is obtained by using a tight cropping of the object to be described.
 
#Cons:
#Requires a very precise segmentation of the object to be described, which is often hard in the real world.
#Normally only used for simple 2D shapes — as shapes become more complex, Hu Moments are not often used.
#Hu Moment calculations are based on the initial centroid computation — if the initial centroid cannot be repeated for similar shapes,
#then Hu Moments will not obtain good matching accuracy.
 