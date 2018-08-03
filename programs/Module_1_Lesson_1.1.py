#Module:1
#Chapter:1
#1.1: Loading, displaying, and saving images

#usage:
#python Module_1_Lesson_1.1.py -i ./images/grand_canyon.png -o ./images/grand_canyon.jpg

#Import necessary packages:
import cv2
import numpy as np
import argparse

#Construct argument parser to parse the command line arguments
#especially the image
ap = argparse.ArgumentParser()

#Add an argument called -i, and make it a mandatory argument
#Add another argument -o, and make it optional
ap.add_argument("-i","--image",required=True,help="Path to the input image")
ap.add_argument("-o","--output",required=False,help="Path to the output image")

#Read the arguments:
args = vars(ap.parse_args())

#args will be a dictionary, and the value of the key "image" will be the 
#path to image

#Read the image:
image = cv2.imread(args['image'])

#Images in opencv are represented as numpy arrays.
#For colored images with RGB color space (more on the color spaces later)
#will have 3 Dimensions. The first dimension is x-axis (or horizontal axis),
#the y-axis (or vertical axis) and the z-axis (for color channels)
#The image origin begins at the left upper corner, and as we move towards right,
#the x-value will increase, and as we move down the y-value will increase.
#For colored images, the third dimension will have colours (pixel intensities)
#in the order BGR (Blue, Green, Red). Each of these values will be between 
#0 to 255. If all three channels have 0, then the pixel is black,
#and if all are 255 then the pixel is white.
#VERY IMPORTANT NOTE:
#We can treat the image in opencv using the x, y axis or using numpy 
#method of slicing/dicing
#In numpy world, the dimensions will be represented in the form RowsXColumnsXDepth
#where Rows = number of rows, Columns = number of columns and Depth = number of elements 
#in the third dimension.
#So, the x-axis of the image will be the number of columns (or image.shape[1]) of the numpy matrix
#The y-axis of the image will be the number of rows of the image (or image.shape[0]).
#The color channel will be image.shape[2]


#Display the dimensions of the image:

print("Width or x-Axis of image (number of columns of numpy array): {}".format(image.shape[1]))
print("Height of y-axis of image (number of rows of numpy array): {}".format(image.shape[0]))
print("Depth of image (for color channels): {}".format(image.shape[2]))

#To display the image:
#The first parameter is a string, or the "name" of our window. 
#The second parameter is a reference to the image we loaded off disk
cv2.imshow("original", image)


#The cv2.waitKey(0) pauses the execution of the script until we press 
#a key on our keyboard. Using a parameter of "0" 
#indicates that any keypress will un-pause the execution.
#cv2.waitKey(1000) will display the image for 1000 milli seconds
#A value of 0 will display the image indefinitely, until any key is pressed
cv2.waitKey(0)

#To save the image in jpg format:
#Note that we are supplying the image format in the name itself,
#and the image will be saved in that format accordingly
#OpenCV will automatically convert the png to jpg

#If we supply -o option then the args["output"] will not be None, else it will be None
if args["output"]:
    cv2.imwrite(args["output"], image)   
    print("Image saved as {}".format(args["output"]))


#To destroy all active windows of cv2.imshow(), use cv2.destroyAllWindows()
cv2.destroyAllWindows()
    