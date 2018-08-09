#Module:1
#Chapter: 1.8: Lighting and color spaces

###########################
##  Lighting Conditions  ##
###########################
#Believe it or not, the success of (nearly) all computer vision systems and applications is determined
#before the developer writes a single line of code. Images taken in poor lighting conditions are really difficult to
#identify.  Lighting conditions play a dramatic role in the successful development of a computer vision system
#Every single computer vision algorithm, application, and system ever developed and that will be developed will depend on the
#quality of images input to the system.
#Lighting can mean the difference between success and failure of your computer vision algorithm,
#and it is "the most important factor" to determine the success of an algorithm.
 
#For instance if you take a picture of your self (in the mirror), standing in front of the mirror and flash on, then the flash
#will interfere with the image, and it is almost impossible to remove the effect of the flash reflected on the image.
 
#One of the most common pitfalls a computer vision developer make is overlooking lighting
#and the effect it will have on algorithm performance.
 
#The camera is not actually “filming” the object itself. Instead, it is capturing the light reflected from our object.
#So when you take a picture of yourself standing in front of a mirror, the camera has captured the flash also coming out of the flash.
 
#In general, your lighting conditions should have three primary goals.
 
#1. High Contrast
#Maximize the contrast between the Regions of Interest in your image
#(i.e. the “objects” you want to detect, extract, describe classify, manipulate, etc.
#should have sufficiently high contrast from the rest of the image so they are easily detectable).
 
#2. Generalizable
#Your lighting conditions should be consistent enough that they work well from one “object” to the next.
#If our goal is to identify various United States coins in an image, our lighting conditions should be
#generalizable enough to facilitate in the coin identification, whether we are examining
#a penny, nickel, dime, or quarter.
#
 
#3. Stable
#Having stable, consistent, and repeatable lighting conditions is the holy grail of computer vision application development.
#However, it’s often hard (if not impossible) to guarantee — this is especially true if we are developing computer vision algorithms
#that are intended to work in outdoor lighting conditions. As the time of day changes, clouds roll in over the sun, and rain starts to pour,
#our lighting conditions will obviously change.
#For example if you are developing a face detection algorithm for a mobile app, then we do not have any control on the lighting conditions
#of the image uploaded by the app users. But if you are developing a OCR system (extracting text from documents in your company), then
#you can control lighting conditions while scanning the documents and make the lighting conditions stable for all the document images.
 
 
#####################################
##  Color Spaces and Color Models  ##
#####################################
#A color space is just a specific organization of colors that allow us to consistently represent and reproduce colors.
#A color model, on the other hand, is an abstract method of numerically representing colors in the color space.
#As a whole, a color space defines both the color model and the abstract mapping function used to define actual colors.
#Selecting a color space also informally implies that we are selecting the color model.
 
#RGB (Red, Green, Blue) color space:
#To define a color in the RGB color model, all we need to do is define the amount of Red, Green, and Blue contained in a single pixel
#Each Red, Green, and Blue channel can have values defined in the range [0, 255] (for a total of 256 “shades”),
#where 0 indicates no representation and 255 demonstrates full representation.
#The RGB color space is an example of an additive color space: the more of each color is added, the brighter the pixel becomes and the closer it comes to white
#R+G = Yellow
#R+B = Pink
#B+G = Magenta
#R+B+G = White
 
#RGB is primarily used to display the contents of the image on monitor, but it is not much suitable for developing computer vision applications
#It is very difficult to determine the actual R, G, B values needed to find the desired color shade.
#But despite how unintuitive the RGB color space may be, nearly all images you’ll work with will be represented (at least initially) in the RGB color space.
 
#Since an RGB color is defined as a 3-valued tuple, with each value in the range [0, 255],
#we can thus think of the cube containing 256 x 256 x 256 = 16,777,216 possible colors,
#depending on how much Red, Green, and Blue we place into each bucket.
 
#We can split an image into RGB channels using:
#b, g, r = cv2.split(image)
#Observe that the first channel is blue, then green and finally red.
 
#import the necessary packages
import argparse
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the original image and display it (RGB)
image = cv2.imread(args["image"])
cv2.imshow("RGB", image)

# loop over each of the individual channels and display them
for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
                cv2.imshow(name, chan)

# wait for a keypress, then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()

#HSV Color space
#The HSV color space transforms the RGB color space, remodeling it as a cylinder rather than a cube
#HSV = Hue, Saturation, Value
#Hue: Which “pure” color we are examining. For example, all shadows and tones of the color “red” will have the same Hue.
#Saturation: How “white” the color is. A fully saturated color would be “pure,” as in “pure red.” And a color with zero saturation would be pure white.
#Value: The Value allows us to control the lightness of our color. A Value of zero would indicate pure black, whereas increasing the value would produce lighter colors.
 
#It’s important to note that different computer vision libraries will use different ranges to represent each of the Hue, Saturation, and Value components.
#However, in the case of OpenCV, images are represented as 8-bit unsigned integer arrays.
#Thus, the Hue value is defined the range [0, 179]. Hue is the actual degree on the HSV color cylinder.
#And both saturation and value are defined on the range [0, 255]
 
#The HSV color space is used heavily in computer vision applications — especially if we are interested in tracking the color of some object in an image.
#It’s far, far easier to define a valid color range using HSV than it is RGB.
 
#To convert an image to HSV, we will use cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
#convert the image to the HSV color space and show it
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

# loop over each of the individual channels and display them
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
                cv2.imshow(name, chan)

# wait for a keypress, then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
 
h,s,v  = cv2.split(hsv)
cv2.imwrite("s.png",s)
 
 
#L*A*B
#While the RGB color space is easy to understand (especially when you’re first getting started in computer vision),
#it’s non-intuitive when defining exact shades of a color or specifying a particular range of colors.
#On the other hand, the HSV color space is more intuitive but does not do the best job in representing how humans see and interpret colors in images.
 
#In RGB and HSV color spaces the Euclidean distance between two arbitrary colors does not have any perceptual meaning.
#But the Euclidean distance in L*a*b* color space has actual perceptual meaning.
#L*a*b* is used heavily in CV. This is due to the distance between colors having an actual perceptual meaning,
#allowing us to overcome various lighting condition problems. It also serves as a powerful color image descriptor.
 
#The addition of perceptual meaning makes the L*a*b* color space less intuitive and understanding as RGB and HSV, but it is heavily used in computer vision.
 
#The L*a*b* color space is a 3-axis system:
#L-channel: The “lightness” of the pixel. This value goes up and down the vertical axis, white to black, with neutral grays at the center of the axis.
#a-channel: Originates from the center of the L-channel and defines pure green on one end of the spectrum and pure red on the other.
#b-channel: Also originates from the center of the L-channel, but is perpendicular to the a-channel.
#The b-channel defines pure blue at one of the spectrum and pure yellow at the other.
 
#To convert an image to L*a*b, we can use the statement:
#cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
 
#cv2.split(image) will split an L*a*b image to L, a, b channels
 
# convert the image to the L*a*b* color space and show it
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)

# loop over each of the individual channels and display them
for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
                cv2.imshow(name, chan)

#wait for a keypress, then close all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
 
#Personally I used the following code block to extract text (OCR) from a document which has
#fields in pre-defined locations on the document.
#I used l channel of L*a*b to remove the background gray lines, followed by the erosion.
#This transformation gave me good result to extract text from image.

l,a,b = cv2.split(lab)
print(l[28:35,28:29])
l[l >= 120] = 255
#l[28:35,28:30] = 0
cv2.imshow("l",l)
cv2.waitKey(0)

cv2.imshow("l_erode",cv2.erode(l,None,1))
cv2.waitKey(0)

#Grayscale
#The last color space we are going to discuss isn’t actually a color space — it’s simply the grayscale representation of a RGB image.
#A grayscale representation of an image throws away the color information of an image and can also be done using the cv2.cvtColor  function:

# show the original and grayscale versions of the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
 
#The grayscale representation of an image is often referred to as “black and white,” but this is not technically correct.
#Grayscale images are single channel images with pixel values in the range [0, 255] (i.e. 256 unique values).
#True black and white images are called binary images and thus only have two possible values: 0 or 255 (i.e. only 2 unique values).
#The grayscale representation of an image is often used when we have no use for color
#(such in detecting faces or building object classifiers where the color of the object does not matter).
#Discarding color thus allows us to save memory and be more computationally efficient.
#Converting an RGB image to grayscale is not as straightforward
#Biologically, our eyes are more sensitive and thus perceive more green and red than blue.

#So while converting RGB to gra image, we use the following equation (to weigh the color channels)
#0.299 R + 0.587 G + 0.114 B
#But I applied the above formula to manually convert a colored image to gray scale, and it did not work.
#May be I made some mistake in calculating the product and rounding the numbers

#Summary:
#1. In general, you’ll find that it’s easier to control your lighting conditions than to write code that compensates for images captured under poor quality.
#2. The RGB color space is the most common color space in computer vision.
#   It’s an additive color space, where colors are defined based on combining values of red, green, and blue.
#   While quite simple, the RGB color space is unfortunately unintuitive for defining colors as it’s hard to pinpoint exactly how much red, green, and blue
#   compose a certain color — imagine looking a specific region of a photo and trying to identify how much red, green, and blue there is using only your naked eye!
#3. Luckily, we have the HSV color space to compensate for this problem.
#   The HSV color space is also intuitive, as it allows us to define colors along a cylinder rather than a RGB cube.
#   The HSV color space also gives lightness/whiteness its own separate dimension, making it easier to define shades of color.
#4. However, both the RGB and HSV color spaces fail to mimic the way humans perceive color — there is no way to mathematically define how
#   perceptually different two arbitrary colors are using the RGB and HSV models. And that’s exactly why the L*a*b* color space was developed.
#   While more complicated, the L*a*b* provides with perceptual uniformity, meaning that the distance between two arbitrary colors has actual meaning.
#5. You will use the RGB color space for most computer vision applications.
#   While it has many shortcomings, you cannot beat its simplicity — it’s simply adequate enough for most systems.
#6. There will also be times when you use the HSV color space — particularly if you are interested in tracking an object in an image based on its color.
#   It’s very easy to define color ranges using HSV.
#7. For basic image processing and computer vision you likely won’t be using the L*a*b* color space that often.
#   But when you’re concerned with color management, color transfer, or color consistency across multiple devices, the L*a*b* color space will be your best friend.
#   It also makes for an excellent color image descriptor.


#Q. As the size of a blurring kernel increases…
#A. The image will appear to be more blurred.
 
#Q. The difference between simple average blurring and Gaussian blurring is…
#A. A Gaussian blur is a weighted average of the local pixels and the average blur is not.
 
#Q. The median blur is appropriate for…
#A. Reducing salt-and-pepper noise.
 
#Q. You would use bilateral filtering when…
#A. When you wanted to smooth your image, but still preserve edges.
 
#Q. The success of a computer vision app starts before a single line of code is written. (Related to lighting)
#A. True
 
#Q. The RGB color space is an example of an additive color space.
#A. True
 
#Q. The HSV color space is less intuitive to define color ranges than RGB.
#A. False
 
#Q. When using the L*a*b* color space, the Euclidean distance between colors has actual perceptual meaning.
#A. True
 
#Q. Convert the following RGB triplet to grayscale: (156, 107, 81)
#A. 0.299*156 + 0.587*107 + 0.114*81 = 119