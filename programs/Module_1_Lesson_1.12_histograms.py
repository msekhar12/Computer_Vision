#Module:1
#Chapter: 1.12 Histograms

#Histograms capture frequency distribution of data.

#A histogram represents the distribution of pixel intensities (whether color or gray-scale) in an image. 
#It can be visualized as a graph (or plot) that gives a high-level intuition of the intensity (pixel value) distribution. 
#We are going to assume a RGB color space in this example, so these pixel values will be in the range of 0 to 255.

#When plotting the histogram, the X-axis serves as our “bins.” If we construct a histogram with 256 bins, 
#then we are effectively counting the number of times each pixel value occurs. In contrast, if we use 
#only 2 (equally spaced) bins, then we are counting the number of times a pixel is in the range [0, 128] or [128, 255]. 
#The number of pixels binned to the x-axis value is then plotted on the y-axis.

#What is the difference between histogram and a bar-chart?
#Histograms plot quantitative data with ranges of the data grouped into bins or intervals while bar charts plot categorical data.

#By simply examining the histogram of an image, you get a general understanding regarding the contrast, brightness, and intensity distribution.
#As per my understanding a histogram shows contrast of an image since a peak in the histogram represents a concentration of pixels with specific values in that peak,
#and these pixels can be easily distinguished from other pixels. A histogram also shows how bright the image is. For example if we have many pixels present 
#around 225 to 255 pixel values, the the image should be a bright image. Histogram obviously shows the intensity distribution, since the intensity of the image
#is represented by its pixel values.

#We will be using the cv2.calcHist function to build our histograms. 

#cv2.calcHist(images, channels, mask, histSize, ranges)
#images: This is the image that we want to compute a histogram for. Wrap it as a list: [myImage].
#channels: A list of indexes, where we specify the index of the channel we want to compute a histogram for. 
#To compute a histogram of a grayscale image, the list would be [0].
#To compute a histogram for all three red, green, and blue channels, the channels list would be [0, 1, 2].
#mask: If a mask is provided, a histogram will be computed for masked pixels only. 
#      If we do not have a mask or do not want to apply one, we can just provide a value of None .
#histSize: This is the number of bins we want to use when computing a histogram. 
#          Again, this is a list, one for each channel we are computing a histogram for. 
#          The bin sizes do not all have to be the same. 
#          Here is an example of 32 bins for each channel: [32, 32, 32] .
#ranges: The range of possible pixel values. Normally, this is [0, 256] (this is not a typo — the ending range of the cv2.calcHist
#        function is non-inclusive so you’ll want to provide a value of 256 rather than 255) for each channel, 
#        but if you are using a color space other than RGB [such as HSV], the ranges might be different.)


# import the necessary packages
from matplotlib import pyplot as plt
import argparse
import cv2
 
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, convert it to grayscale, and show it
image = cv2.imread(args["image"])

image_color = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
 
# construct a grayscale histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
 
# plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

#The above code will plot the unnormalized plot (that is it has the actual frequency of the occurrance of each picel). 
#To compare the images, we should not use actual frequency, as two identical images (with different sizes) will have two 
#different frequency distributions (although the shape of the distribution should be identical). If we use normalized distribution,
#then the y-axis should have the percentage pixels in the image having a value in the x-axis.

#To normalize the histogram, use the following statement:
hist /= hist.sum()

# plot the normalized histogram
plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("Bins")
plt.ylabel("% of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()



#Let us plot histogram for each color channel of the image
# grab the image channels, initialize the tuple of colors and the
# figure
image = image_color
chans = cv2.split(image)

#Remember that the channels in opencv will be in the order b, g, r
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
 
# loop over the image channels
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and plot it
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color = color)
	plt.xlim([0, 256])

plt.show()	

	
	
#2D Histogram
#Up until this point, we have computed a histogram for only one channel at a time. 
#Now we move on to multi-dimensional histograms and take into consideration two channels at a time.

#For example, we can ask a question such as: “How many pixels have a Red value of 10 AND a Blue value of 30?” 
#“How many pixels have a Green value of 200 AND a Red value of 130?” 
#By using the conjunctive AND, we are able to construct multi-dimensional histograms.	

# let's move on to 2D histograms -- we need to reduce the
# number of bins in the histogram from 256 to 32 so we can
# better visualize the results
fig = plt.figure()
 
# plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)
 
# plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)
 
# plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)
 
# finally, let's examine the dimensionality of one of the 2D
# histograms
print("2D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

plt.show()

#Using a 2D histogram takes into account two channels at a time. 
#But what if we wanted to account for all three RGB channels? We’re now going to build a 3D histogram.	
# our 2D histogram could only take into account 2 out of the 3
# channels in the image so now let's build a 3D color histogram
# (utilizing all channels) with 8 bins in each direction -- we
# can't plot the 3D histogram, but the theory is exactly like
# that of a 2D histogram, so we'll just show the shape of the
# histogram
hist = cv2.calcHist([image], [0, 1, 2],
	None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))
 
# Show our plots
plt.show()


#Histogram Equalization
#Histogram equalization improves the contrast of an image by “stretching” the distribution of pixels. 
#Consider a histogram with a large peak at the center of it. Applying histogram equalization will stretch the peak 
#out towards the corner of the image, thus improving the global contrast of the image. 
#Histogram equalization is applied to grayscale images.

#This method is useful when an image contains foregrounds and backgrounds that are both dark or both light. 
#It tends to produce unrealistic effects in photographs; however, is normally useful when enhancing the contrast of medical or satellite images.

#Performing histogram is very simple.
#We will use the following statement:
#eq = cv2.equalizeHist(image)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# apply histogram equalization to stretch the contrast of our image
eq = cv2.equalizeHist(image)
 
# show our images -- notice how the contrast of the second image has
# been stretched
cv2.imshow("Original", image)
cv2.imshow("Histogram Equalization", eq)
cv2.waitKey(0)

#Check how the distribution of pixel values changed between image and equalized image:
plt.figure()
plt.title("equalized Histogram")
plt.xlabel("Bins")
plt.ylabel("% of Pixels")

hist = cv2.calcHist([eq], [0], None, [256], [0, 256])

#To normalize the histogram, use the following statement:
hist /= hist.sum()

plt.plot(hist, color = color)
plt.xlim([0, 256])

plt.show()	

#Check how the distribution of pixel values changed between image and equalized image:
plt.figure()
plt.title("non-equalized Histogram")
plt.xlabel("Bins")
plt.ylabel("% of Pixels")

hist = cv2.calcHist([image], [0], None, [256], [0, 256])

#To normalize the histogram, use the following statement:
hist /= hist.sum()

plt.plot(hist, color = color)
plt.xlim([0, 256])

plt.show()	

#Create a helper function to plot histograms:
# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import cv2
 
def plot_histogram(image, title, mask=None, normalize=False):
	# grab the image channels, initialize the tuple of colors and
	# the figure
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    if normalize:
       plt.ylabel("% of Pixels")
    else:   
        plt.ylabel("# of Pixels")
 
	# loop over the image channels
    for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        if normalize:
            hist /= hist.sum()
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()    

    
    
def plot_2D_histogram(image, bins=32):    
    chans = cv2.split(image)
    fig = plt.figure()
    # plot a 2D color histogram for green and blue
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [bins, bins],
	       [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for G and B")
    plt.colorbar(p)
 
    # plot a 2D color histogram for green and red
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [bins, bins],
	       [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for G and R")
    plt.colorbar(p)
 
    # plot a 2D color histogram for blue and red
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [bins, bins],
	       [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for B and R")
    plt.colorbar(p)
 
    # finally, let's examine the dimensionality of one of the 2D
    # histograms
    print("2D histogram shape: {}, with {} values".format(
	    hist.shape, hist.flatten().shape[0]))

    plt.show()
        
        
        
#Questions:
#Use images/horseshoe_bend.png to answer

#1. Compute a grayscale histogram for the image. Approximately what bin number has the highest pixel count?
#Ans: 46. Use code given below:

image = cv2.imread("./images/horseshoe_bend.png")

cv2.imshow("original", image)
cv2.waitKey(0)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", image_gray)
cv2.waitKey(0)
plot_histogram(image_gray, title="horseshoe", mask=None, normalize=False)

#2. Compute a flattened RGB histogram for our image. Which of the following is true:
#a. The blue bin count is greater than the green bin count for bin #200.
#b. The red bin count is larger than the green bin count for bin #200.
#c. The blue bin count is less than the red bin count for bin #250.
#d. The green bin count is greater than the blue bin count for bin #100.
#Ans: b. See below code
image = cv2.imread("./images/horseshoe_bend.png")
plot_histogram(image, title="horseshoe", mask=None, normalize=False)


#3. Compute a 2D color histogram for the Blue and Red channels of our image using 32 x 32 bins. Which bins have the largest pixel count?
#A. x=4, y=5. See code below
plot_2D_histogram(image, bins=32)

#4. Perform histogram equalization on our image. After performing histogram equalization, what is the value of the pixel located at x=146, y=272?
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# apply histogram equalization to stretch the contrast of our image
eq = cv2.equalizeHist(image)
print(eq[272, 146])


#5. Compute a 3D color histogram using 8 bins for the Red channel, 16 bins for the Green channel, and 9 bins for the Blue channel. What is the total # of bins in our histogram?
#Ans: 1152