#Module:1
#Chapter:1.2
#1.2: Image Basics

#usage:
#python Module_1_Lesson_1.2.py -i ./images/grand_canyon.png -o ./images/grand_canyon.jpg

#pixel: Normally, a pixel is considered the “color” or the “intensity” of light that appears in a given place in our image.
#       Pixels are the raw building blocks of an image. Every image consists of a set of pixels. There is no finer granularity than the pixel.
#Most pixels are represented in two ways: grayscale and color. 
#In a grayscale image, each pixel has a value between 0 and 255, where zero is corresponds to 'black and 255 being 'white'. 
#The values in between 0 and 255 are varying shades of gray, where values closer to 0 are darker and values closer to 255 are lighter.

#Color pixels, however, are normally represented in the RGB color space — one value for the Red component, one for Green, and one for Blue, 
#leading to a total of 3 values per pixel

#Each of the three Red, Green, and Blue colors are represented by an integer in the range 0 to 255, 
#which indicates how 'much' of the color there is. 
#Given that the pixel value only needs to be in the range [0, 255] we normally use an 8-bit unsigned integer to represent each color intensity.
#To construct a white color, we would fill each of the red, green, and blue buckets completely up, like this: (255, 255, 255), since white is the presence of all color.
#To create a black color, we would empty each of the buckets out: (0, 0, 0), since black is the absence of color.

#It’s important to note that OpenCV stores images in BGR (Blue, Green, Red) order rather than RGB order since this caveat could cause some confusion later.

#To get the B, G, R values of a pixel at image[0,0], we can say:
#B, G, R = image[0:0]

#To make a pixel value as red:
#image[0:0] = (0,0,255)

#To set top right corner of the image (quarter) as green:
#width = image.shape[1]
#height = image.shape[0]
#image[0:height//2, width//2:] = (0, 255, 0)

#Read an image:
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="path to image")

args = vars(ap.parse_args())
image = cv2.imread(args['image'])

height, width = image.shape[:2]
#image[0:height//2, width//2:] = (0, 255, 0)

cv2.imshow("image",image)
cv2.waitKey(0)

print(image[225, 111])