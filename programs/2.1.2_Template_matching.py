#2.1.2: Template matching
 
#Template matching is both extremely simple and practically effortless but has a major drawback of only working under very specific (or highly controlled) conditions.
 
#Before we can even apply template matching, we need to gather two images:
#Source image: This is the image we expect to find a match to our template in.
#Template image: The “object patch” we are searching for in the source image.
 
#To find the template in the source image, we slide the template from left-to-right and top-to-bottom across the source.
 
#Let the source Image is represented as I, and its dimensions as iW X iH
#Let the template Image is represented as T, and its dimensions as tW X tH
#For the Template matching to work the T must fit inside the I.
#So iW >= tw and iH >= tH
 
#In cv2, you can use the following function to match template:
#cv.MatchTemplate(image, templ, result, method)
#where image = I
#      templ = T
#      result = a matrix having the metric values computed as we slide the T over I
#      method = comparison methods (see https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html)
 
#result will be a single channel with (iW - tW + 1) X (iH - tH +1) Size
#The function slides through image , compares the overlapped patches of size width and height against templ using the specified
#method and stores the comparison results in result
 
#While template matching is extremely simple to apply, it’s only useful in a very limited set of circumstances.
#In nearly all cases, you’ll want to ensure that the template you are detecting is nearly identical to the object you want
#to detect in the source. Even small, minor deviations in appearance can dramatically affect the results from
#template matching and render it effectively useless.
 
# import the necessary packages
import argparse
import cv2
import numpy as np
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to the source image")
ap.add_argument("-t", "--template", required=True, help="Path to the template image")
args = vars(ap.parse_args())

# load the source and template image
source = cv2.imread(args["source"])
template = cv2.imread(args["template"])
(tempH, tempW) = template.shape[:2]

# find the template in the source image
# The cv2.matchTemplate  method requires three parameters: the source  image, the template , and the template matching method.
# We’ll pass in cv2.TM_CCOEFF  to indicate we want to use the correlation coefficient method.
result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
 
# Now that we have computed the result  matrix, we can apply the cv2.minMaxLoc  function to find the (x, y) coordinates of the best match.
# We pass in our result, and in return we receive a 4-tuple consisting of the minimum value in the result,
# the maximum value in result, the (x, y) coordinates of the minimum value, and the (x, y) coordinates of the maximum value, respectively
 
(minVal, maxVal, minLoc, (x, y)) = cv2.minMaxLoc(result)
print("minVal:{}, maxVal: {}, minLoc: {}, (x: {}, y: {})".format(minVal, maxVal, minLoc, x, y))
 
cv2.imshow("source", source)
cv2.imshow("template", template)
cv2.waitKey(0)
 
# Draw a rectangle on the image, where the template has matched:
cv2.rectangle(source, (x,y), (x+tempW,y+tempH), (0,255,0), 2)
cv2.imshow("found", source)
cv2.waitKey(0)
 
#But the above code does not work if the sizes of image do not match or if the
#object shape is a bit different or if there is some noise in the image.
 
 
#To perform template matching on different sizes (template size and the object size inside the image are different),
#Adrian has written a wonderful article at https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
 
#The following notes is prepared using that article.

#To perform mult-scale template match, we resize the image at various scales (till the image is not less than the template size),
#and perform the template matching on each resized image. In this process, the ratio of original image's height and resized image's height is stored.
#Steps:
#1. Convert template to gray and apply canny edge dector
#2. Best_match = None
#3. For each resized_image:
#         if resized_image is smaller than template, then break
#         else:
#            3a. Let resized_image is the current resized image
#            3b. r = (obtain original image's height)/(resized_image_height)
#            3c. Convert resized image to gray and apply canny edge dector
#            3d. temp_best_match = Obtain the template's best match parameters
#            3e. if Best_Match is None or if the temp_best_match is better than Best_Match:
#                 Update Best_Match = temp_best_match
#     
#  
 
 
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
 
#convert template to gray scale
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Template", template)
 
#apply canny edge detector to template
#for canny edge detector...see https://github.com/msekhar12/Computer_Vision/blob/master/programs/Module_1_Lesson_1.10.2_edge_detection.py
template = cv2.Canny(template, 50, 200)
 
#Get width and height of template
(tH, tW) = template.shape[:2]
 
#display the template
cv2.imshow("Template", template)
cv2.waitKey(0)
#read the source image
image = cv2.imread(args["source"])
 
#convert the source to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 

#initialize the parameter
found = None
 
#side note:
#np.linspace(0.2, 1.0, 20)[::-1] will generate 20 equi-spaced real numbers between [0.2, 1.0] and orders thme in descending order
#np.linspace(0.2, 1.0, 20) will generate 20 equi-spaced real numbers between [0.2, 1.0] and orders thme in ascending order
 
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
            break
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (ignore_1, maxVal, ignore_2, maxLoc) = cv2.minMaxLoc(result)
    print(ignore_1, maxVal, ignore_2, maxLoc)
    print("found: {}".format(found))
    # if we have found a new maximum correlation value, then ipdate
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            print("found: {}".format(found))

                # unpack the bookkeeping varaible and compute the (x, y) coordinates
                # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
 
# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
cv2.imshow("Image", imutils.resize(image, width=600))
cv2.waitKey(0)           


 
#See https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
#for a detailed explanation for multi-size template matching.   
 
#Q. In the context of template matching, the template image is:
#A. The “object” we are trying to find in new images.
 
#Q. Therefore, our source image is:
#A. The image containing the “object we want to find”.
 
#Q.The primary limitation of template matching is:
#A.  It’s hard to implement.
#B.  It’s an extremely slow algorithm.
#C.  It only works for templates that have corresponding identical (or near identical) patches in the source image.
#D.  It operates on the raw pixel intensities of images.
#Answer: C
 
#Q. Find the (x,y) location where the image template_q.png is matched in source_q.png.
#A.
source = cv2.imread("./images/source_q.png")
template = cv2.imread("./images/template_q.png")
 
cv2.destroyAllWindows()
cv2.imshow("source", source)
cv2.imshow("template", template)
result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
(minVal, maxVal, minLoc, (x, y))= cv2.minMaxLoc(result)
print("(x: {}, y:{})".format(x,y))
#291, 1680
cv2.waitKey(0)