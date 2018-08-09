#Module:1
#Chapter: 1.11: Contours
#1.11.3: Advanced contour properties
 
#In 1.11.2 we learned about basic contour properties. Using basic contour properties, we can
#create more advanced contour properties like aspect ratio, extent, convex hull, and solidity
#The more advanced properties allow us to discriminate between and recognize various shapes in images
 
#Advanced Contour Properties
#Contour properties can take you a long, long way when building computer vision applications, especially when you are
#first getting started. It takes a bit of creative thinking and a lot of discipline not to jump to more advanced techniques
#such as machine learning and training your own object classifier, but by paying attention to contour properties we can
#actually perform object identification for simple objects quite easily.
 
################
#Aspect Ratio  #
################
 
#aspect ratio = image width / image height
#Aspect ratio can be used to distinguish between squares and rectangles and detect handwritten digits in images
#and prune them from the rest of the contours.
 
#Shapes with an aspect ratio < 1 have a height that is greater than the width. These shapes will appear to be more “tall” and elongated.
#For example, most digits and characters on a license plate have an aspect ratio that is less than 1
#(since most characters on a license plate are taller than they are wide).
 
#And shapes with an aspect ratio > 1 have a width that is greater than the height.
#The license plate itself is an example of a object that will have an aspect ratio greater than 1 since the
#width of a physical license plate is always greater than the height
 
#Shapes with an aspect ratio = 1 (plus or minus some error of course), have approximately the same width and height.

#Squares and circles are examples of shapes that will have an aspect ratio of approximately 1.

#NOTE: As per my understanding the aspect ratio is calculated using the bounding rectangle's width and height.
 
##########
#Extent  #
##########
 
#The extent of a shape or contour is the ratio of the contour area to the bounding box area.
 
#extent = shape area / bounding box area
 
#The area of an actual shape is simply the number of pixels inside the contoured region.
#The rectangular area of the contour is determined by its bounding box
#The bounding box area = bounding box width X bounding box height
 
#The "extent" will always be less than 1, as the bounding box's area is always greater than or equal to the area of the encloded object
#Whether or not you use the extent when trying to distinguish between various shapes in images is entirely dependent on your application.
#And furthermore, you’ll have to manually inspect the values of the extent to determine which ranges are good for distinguishing between shapes
 
 
###############
#Convex Hull  #
###############
#If you wrap a super elastic rubberband on a shape, you get what is called convex hull.
#This super elastic rubber band never leaves any extra space or any extra slack. It requires the minimum amount of space to enclose points of an object.
#Thus the convex hull is the minimum enclosing polygon of all points of the input shape.
#Convexity defects: It is an important concept of convex hull.
#Convex curves are curves that appear to “bulged out”.
#If a curve is not bulged out, then we call it a "convexity defect".
#For example if we draw a convex hull around the outline of the hand (the fingers spread), then the convex hull between the fingets is not bulged out.
#The convex hull and convexity defects play a major role in hand gesture recognition,
#as it allows us to utilize the convexity defects of the hand to count the number of fingers.
 
###########
#Solidity #
###########
#The last advanced contour is the solidity of a shape.
#The solidity of a shape is the area of the contour area divided by the area of the convex hull:
 
#solidity = contour area / convex hull area
 
#solidity is always <= 1
 
#Just as in the extent of a shape, when using the solidity to distinguish between various objects you’ll need to manually inspect the values of the solidity
#to determine the appropriate ranges. For example, the solidity of a shape is actually perfect for distinguishing between the X’s and O’s on a tic-tac-toe board.
 
###########################################
#Advanced contour properties applications #
###########################################
 

#i. Identify X and O symbols from tictactoe game
 
#We will use the solidity concept to distinguish between "O" nd "X" characters on a tictactoe game. See tittactoe.png image in the images folder.
#We will use that image to identify O and X symbols.
 
# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

# load the tic-tac-toe image and convert it to grayscale
image = cv2.imread("images/tictactoe.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find all contours on the tic-tac-toe board
#We have to use cv2.RETR_EXTERNAL. If not used, the O character will be identified as 2 contours (due to hollowness in the character)
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Check the version of cv2 and obtain the contours
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for (i, c) in enumerate(cnts):
                # compute the area of the contour along with the bounding box
                # to compute the aspect ratio
                area = cv2.contourArea(c)
                (x, y, w, h) = cv2.boundingRect(c)

                # compute the convex hull of the contour, then use the area of the
                # original contour and the area of the convex hull to compute the
                # solidity
                hull = cv2.convexHull(c)
                hullArea = cv2.contourArea(hull)
                solidity = area / float(hullArea)
   
                # initialize the character text
                char = "?"

                # if the solidity is high, then we are examining an `O`
                if solidity > 0.9:
                                char = "O"

                # otherwise, if the solidity it still reasonabably high, we
                # are examining an `X`
                elif solidity > 0.5:
                                char = "X"

                # if the character is not unknown, draw it
    #That is, draw a contour if and only if the character is determined as X or O
                if char != "?":
                                cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
                                cv2.putText(image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                                                (0, 255, 0), 4)
                # show the contour properties
                print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
 
#The values of 0.9 (to identify O) and 0.5 (to identify X) were obtained by displaying the
#solidity values of the shapes. For the grid the solidity value is 0.28, and hence
#the grid is not contoured on the image.
 
#ii. Identifying Tetris Blocks
#Distinguishing between X’s and O’s in a tic-tac-toe game is a great introduction to the power of contour properties.
#But in the previous example we used only the solidity perform our identification. In some cases, such as in identifying the various
#types of Tetris blocks, we need to utilize more than one contour property. Specifically, we’ll be using aspect ratio, extent, convex hull,
#and solidity in conjunction with each other to perform our brick identification.
#For this case-study we will use tetris_blocks.png image
 
#Read the image
image = cv2.imread("./images/tetris_blocks.png")
cv2.imshow("original", image)
cv2.waitKey(0)
 
#Convert to Gray scaled image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#If the pixel intensity is less than 225, then make it 255, else make it 0
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("threshold", thresh)
cv2.waitKey(0)
 
# find external contours in the thresholded image and allocate memory
# for the convex hull image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
hullImage = np.zeros(gray.shape[:2], dtype="uint8")
 
#Loop over the contours and display the advanced properties of each of the objects, so that
#we can identify how to distinguish between the objects
clone = image.copy()
for (i, c) in enumerate(cnts):
    #Get the centroid and add the number of the shape,
    #so that we can associate the number with the display
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    # compute the area of the contour along with the bounding box
                # to compute the aspect ratio
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)
    # compute the aspect ratio of the contour, which is simply the width
                # divided by the height of the bounding box
    aspectRatio = w / float(h)

                # use the area of the contour and the bounding box area to compute
                # the extent
    extent = area / float(w * h)

                # compute the convex hull of the contour, then use the area of the
                # original contour and the area of the convex hull to compute the
                # solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)
    print("For the shape ID:{}".format(i+1))
    print("aspectRatio: {}, extent: {}, solidity: {}".format(aspectRatio, extent,solidity))
 
cv2.imshow("clone",clone)   
cv2.waitKey(0)
 
#The above program has identified the objects as follows:
#ID 1 is identified as a Z shaped block (red colored)
#ID 2 is identified as a Z shaped block, but inverted (or reflected on the y-axis) (green colored)
#ID 3 is identified as the rectangle block (blue colored)
#ID 4 is identified as square (yellow colored)
#ID 5 is identified as L shaped object (orange)
#ID 6 is identified as L shaped object (blue)
 
#Obtained the following contour properties:
 
#For the shape ID:1
#aspectRatio: 1.4833333333333334, extent: 0.6569288389513108, solidity: 0.8118491090025457
#For the shape ID:2
#aspectRatio: 1.4666666666666666, extent: 0.6537878787878788, solidity: 0.804287045666356
#For the shape ID:3
#aspectRatio: 3.757575757575758, extent: 0.9618768328445748, solidity: 1.0
#For the shape ID:4
#aspectRatio: 1.0, extent: 0.9658145065398336, solidity: 1.0
#For the shape ID:5
#aspectRatio: 1.492063492063492, extent: 0.6338230327592029, solidity: 0.788633259796197
#For the shape ID:6
#aspectRatio: 1.492063492063492, extent: 0.6338230327592029, solidity: 0.788633259796197
 
#NOTE:
#For square (or ID 4 object), the aspect ratio is 1. So we can determine the object as square if the aspect ratio is 1.
#If solidity is 1 and aspect ratio is much higher than 1 (in fact near by 3) or much lower than 1 (less than 0.5) then regard that as a rectangle
#If solidity is greater than 0.8, then it should be Z shaped
#If the exient is less than 0.65 then it will be L piece.
#NOTE that the Z shape and L shape have almost nearest values. But it worked in this example using the above values.
#We should test on more images to make sure that our rules are generic enough to identify various shapes.
 
#The complete program to identify tetris shapes is given below:
 
# loop over the contours
# loop over the contours
for (i, c) in enumerate(cnts):
                # compute the area of the contour along with the bounding box
                # to compute the aspect ratio
                area = cv2.contourArea(c)
                (x, y, w, h) = cv2.boundingRect(c)

                # compute the aspect ratio of the contour, which is simply the width
                # divided by the height of the bounding box
                aspectRatio = w / float(h)

                # use the area of the contour and the bounding box area to compute
                # the extent
                extent = area / float(w * h)

                # compute the convex hull of the contour, then use the area of the
                # original contour and the area of the convex hull to compute the
                # solidity
                hull = cv2.convexHull(c)
                hullArea = cv2.contourArea(hull)
                solidity = area / float(hullArea)

                # visualize the original contours and the convex hull and initialize
                # the name of the shape
                cv2.drawContours(hullImage, [hull], -1, 255, -1)
                cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
                shape = ""

                # if the aspect ratio is approximately one, then the shape is a square
                if aspectRatio >= 0.98 and aspectRatio <= 1.02:
                                shape = "SQUARE"

                # if the width is 3x longer than the height, then we have a rectangle
                elif aspectRatio >= 3.0:
                                shape = "RECTANGLE"

                # if the extent is sufficiently small, then we have a L-piece
                elif extent < 0.65:
                                shape = "L-PIECE"

                # if the solidity is sufficiently large enough, then we have a Z-piece
                elif solidity > 0.80:
                                shape = "Z-PIECE"

				# draw the shape name on the image
                cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (240, 0, 159), 2)

                # show the contour properties
                print("Contour #{} -- aspect_ratio={:.2f}, extent={:.2f}, solidity={:.2f}"
                                .format(i + 1, aspectRatio, extent, solidity))

                # show the output images
                cv2.imshow("Convex Hull", hullImage)
                cv2.imshow("Image", image)

                cv2.waitKey(0)
 
cv2.destroyAllWindows()
   
#As you can see, using nothing more than the aspect ratio, extent, and solidity of a shape we were able to distinguish between the four different types of Tetris blocks.
#Using simple contour properties, we were able to recognize X’s and O’s on a tic-tac-toe board.
#And we were also able to recognize the various types of Tetris blocks.
#Again, these contour properties, which are very simple on the surface, can enable us to identify various shapes.
#We just need to take a step back, be a little clever, and inspect the values of each of our contour properties to construct rules to identify each shape.   
 
#Questions:
#For this quiz, you’ll need to download the following image: http://pyimg.co/zua9t (see more_shapes_example.png in images)
#1. Then use advanced contour properties to answer “What is the aspect ratio of the circle?”
#Ans: 1
 
#2. What is the aspect ratio of the orange rectangle?
#Ans: 3.3
 
#What is the solidity of the arrow?
#Ans: 0.78
 
#What about the extent of the purple arrow?
#Ans: 0.48
 
#The convex hull of a shape is:
#Ans: The smallest possible convex set of points that fully contains a given shape.
 
image = cv2.imread("./images/more_shapes_example.png")
cv2.imshow("image",image)
cv2.waitKey(0)
 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
_,cnts,_ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clone = gray.copy()
for (i,c) in enumerate(cnts):
    #Get the centroid and add the number of the shape,
    #so that we can associate the number with the display
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)    
    (x, y, w, h) = cv2.boundingRect(c)
   
    #Finding solidity
    area = cv2.contourArea(c)
    ar = w/float(h)
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)
   
    #Finding extent
    extent = area/float(w*h)
    print("Aspect Ratio of ID: {} is {}, solidity is {} and extent is {}".format(i+1, ar, solidity, extent))
cv2.imshow("clone",clone)

cv2.waitKey(0)   