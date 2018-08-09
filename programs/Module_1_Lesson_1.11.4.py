#Module:1
#Chapter: 1.11: Contours
#1.11.4: Contour approximation
 
#As the name suggests, contour approximation is an algorithm for reducing the number of points in a curve
#with a reduced set of points — thus, an approximation.
#This algorithm is commonly known as the Ramer-Douglas-Peucker algorithm, or simply: the split-and-merge algorithm.
 
#The general assumption of this algorithm is that a curve can be approximated by a series of short line segments.
#And we can thus approximate a given number of these line segments to reduce the number of points it takes to construct a curve.
 
#Overall, the resulting approximated curve consists of a subset of points that were defined by the original curve.
#In openCV we have an inbuilt function called cv2.approxPolyDP(), which finds the contour approximation.
 
#To find more information about the algorithm's implementation see https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
 
#Let us read the image circles_and_squares.png present in images folder, and our goal is to identify only squares and rectangles,
#and ignore circles and ellipses present in the image.
 
#The logic is simple. We will apply contour approximation to each contour and check if the approximated contour
#has 4 sides. If yes, then we will consider that as rectangle, else ignore that object (or contour)
 
#The cv2.approxPolyDP() will take 3 parameters. The first parameter is contour, the second parameter is the permitted error (epsilon).
#To control the level of tolerance for the approximation, we need to define a epsilon value. In practice, we define this epsilon relative to the perimeter
#of the shape we are examining. Commonly, we’ll define epsilon as some percentage (usually between 1-5%) of the original contour perimeter.
#This is because the internal contour approximation algorithm is looking for points to discard. The larger the epsilon value is, the more points will be discarded.
#Similarly, the smaller the epsilon value is, the more points will be kept.
#In other words if the permitted error (expressed as epsilon) is large, more points will be discarded.
#So how do we supply the optimal allowed epsilon (or error)? It depends on the size and shape of the object.
#Thus, we define epsilon relative to the perimeter length so we understand how large the contour region actually is.
#Doing this ensures that we achieve a consistent approximation for all shapes inside the image.
#It is typical to use roughly 1-5% of the original contour perimeter length for a value of epsilon.
 
#The permitted error is expressed as fraction change allowed in the perimeter. So if we supply 0.1*perimeter as allowed error, then
#we are implying that the new perimeter after contour approximation should not change more than 10% of the original object's perimeter.
 
#The third parameter will mention if the contour is closed. True means it is closed
 
#Example: cv2.approxPolyDP(c, 0.01 * peri, True)
 
# import the necessary packages
import cv2
import imutils

# load the the cirles and squares image and convert it to grayscale
image = cv2.imread("images/circles_and_squares.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)

                # if the approximated contour has 4 vertices, then we are examining
                # a rectangle
                if len(approx) == 4:
                                # draw the outline of the contour and draw the text on the image
                                cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
                                (x, y, w, h) = cv2.boundingRect(approx)
                                cv2.putText(image, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, (0, 255, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
 
 
#Another application: Identify the document in a image.
 
# import the necessary packages
import cv2

# load the receipt image, convert it to grayscale, and detect
# edges
image = cv2.imread("images/receipt.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)
 
# show the original image and edged map
cv2.imshow("Original", image)
cv2.imshow("Edge Map", edged)
 
# find contours in the image and sort them from largest to smallest,
# keeping only the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]

# loop over the contours
for c in cnts:
                # approximate the contour and initialize the contour color
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)

                # show the difference in number of vertices between the original
                # and approximated contours
                print("original: {}, approx: {}".format(len(c), len(approx)))

                # if the approximated contour has 4 vertices, then we have found
                # our rectangle
                if len(approx) == 4:
                                # draw the outline on the image
                                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
 
 
#Questions:
#To answer the remaining questions in this quiz, download the following image: http://pyimg.co/1e5ok (see dog_contour.png)
#Then, find the contour of the dog silhouette in the image and approximate the contour using 1% the length of the perimeter.
 
#Q. How many points are in the approximated contour?
#Ans: 17
 
#Q. How many points are in the approximated contour when we use 5% of the original perimeter?
#Ans: 5
 
#Q. How many points are in the approximated contour when we use 10% of the original perimeter?
#Ans: 3
 
#Q. As the percentage of perimeter increases, the number of points in the approximation:
#Ans: Decreases
 
import numpy as np
image = cv2.imread("images/dog_contour.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
_,cnts,_ = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("dog",cv2.drawContours(image, [cnts[0]], -1, (0, 255, 0), 2))
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
peri = cv2.arcLength(cnts[0], True)
approx = cv2.approxPolyDP(cnts[0], 0.05 * peri, True)
dog_contoured = np.ones(gray.shape,dtype="uint8")*255
print(len(cnts[0]))
print(len(approx))
cv2.imshow("dog_contoured",cv2.drawContours(dog_contoured, [approx], -1, (0, 255, 0), 2))
 
cv2.waitKey(0)