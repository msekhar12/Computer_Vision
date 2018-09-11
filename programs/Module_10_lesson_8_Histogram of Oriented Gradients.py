#10.8: Histogram of Oriented Gradients
 
#The main purpose of HOG is Object detection in images.
#They can also be used for representing Texture and Shape
 
#The most important parameters for the HOG descriptor are the orientations, pixels_per_cell, and the cells_per_block.
#These three parameters (along with the size of the input image) effectively control the dimensionality of the resulting feature vector.
 
#HOGs are usually combined with Machine Learning (ML) algorithms to perform Object detection within an image.
#That is, HOGs can be used as features to train ML models.
 
#The reason HOG is utilized so heavily is because local object appearance and shape can be characterized
#using the distribution of local intensity gradients (see gradients lesson of module 1 for more info about gradients)
 
#We will use scikit-image implementation of HOG, although OpenCV also provides methods to get HOGs of an image.
 
#HOG descriptors are mainly used to describe the structural shape and appearance of an object in an image,
#making them excellent descriptors for object classification.
#However, since HOG captures local intensity gradients and edge directions,
#it also makes for a good texture descriptor.
 
#########################################
#Steps for computing the HOG descriptor #
#########################################
 
#######################################################
#Step 1: Normalizing the image prior to description  ##
#######################################################
#You may or may not need this step. But if you do apply, you may consider the following forms of normalization:
#log(p) (aka Gamma/power law normalization)
#sqrt(p) (aka Square-root normalization)
#(p - avg)/sd (aka Variance normalization)
#where avg = avg of all pixels and sd = std. dev of pixel intensities
 
#In most cases, it’s best to start with either no normalization or square-root normalization.
#Variance normalization is also worth consideration, but in most cases it will perform in a
#similar manner to square-root normalization.
 
#################################
#Step 2: Gradient computation  ##
#################################
#See https://github.com/msekhar12/Computer_Vision/blob/master/programs/Module_1_Lesson_1.10.1_gradients.py
#for more infor on gradient computation.
 
#G = sqrt(Gx*Gx + Gy*Gy)
#Theta = TanInverse(Gx, Gy)
#Where G = Gradient, Gx = Gradient in X-direction, Gy = Gradient in Y-direction
#Theta = Orientation or slope of the gradient
 
#Each pixel will have a G and theta.
 
#The bin of the histogram will be theta, and the contribution or weight added to a given bin is based on G
 
#######################################
#Step 3: Weighted votes in each cell ##
#######################################
 
#After calculating magnitudes and orientations of each pixel, the next step is to divide the image into cells and blocks
#A 'cell' is a rectangular region defined by the number of pixels that belong in each cell.
#For example, if we had a 128 x 128 image and defined our pixels_per_cell  as 4 x 4, we would thus have 32 x 32 = 1024 cells
#Since 128/4 = 32 and 128/4 = 32.
#If we divide pixels per cell as 64 x 64, then we will get 2 x 2 = 4 cells
#Since 128/64 = 2 and 128/64 = 2
 
#For each cell, using the G(gradients) and Theta(slope) of each pixel in the cell, we need to construct the histograms of orientations.
#To construct histograms, we need to determine our desired number of orientations. The number of orientations control the
#number of bins in the resulting histogram.
#The gradient angle (theta) is either within the range [0, 180] (unsigned) or [0, 360] (signed).
#If we use unsigned, and use orientations = 9, then the number of bins will be 9. So, we need to divide the 180 with a
#number, such that we get 9 intervals (or bins)
#As per my understanding...
#Assume that we have the following pixels in a cell:
#p1: |G| = 150, theta = 12
#p2: |G| = 1000, theta = 45
#p3: |G| = 50, theta = 120
#p4: |G| = 75, theta = 125
#And we used orientations = 9 (9 bins). So we will have the following intervals:
#[0, 20), [20,40), [40, 60), [60, 80), [80, 100), [100, 120), [120, 140), [140, 160), [160, 180)
#The the interval [0, 20) will have 150
#The the interval [40, 60) will have 1000
#The the interval [120, 140) will have 125 (since 50+75)

#At this point, we could collect and concatenate each of these histograms to form our final feature vector. However, it’s beneficial to apply block normalization (step 4)
##############################################
#Step 4: Contrast normalization over blocks ##
##############################################
#To account for changes in illumination and contrast, we can normalize the gradient values locally.
#This requires grouping the “cells” together into larger, connecting “blocks”.
#It is common for these blocks to overlap, meaning that each cell contributes to the final feature vector more than once.
#
#Example:
#Assume that after dividing our image into cells, we have the following cells:
#C1 C2 C3
#C4 C5 C6
#C7 C8 C9
#
#We will group adjacent cells into blocks. For example, if we use 2 x 2 (cells per block) as the block size, then
#we will get the following blocks as groups:
#BLOCK-1
#C1 C2
#C4 C5
 
#BLOCK-2
#C2 C3
#C5 C6

#BLOCK-3
#C4 C5
#C7 C8
 
#BLOCK-4
#C5 C6
#C8 C9
 
#For each of the cells in the current block we concatenate their corresponding gradient histograms, followed by either L1 or L2 normalizing
#the entire concatenated feature vector.
#Performing this type of normalization implies that each of the cells will be represented in the final feature vector multiple times but
#normalized by a different value. While this multi-representation is redundant and wasteful of space, it actually increases performance of the descriptor.
 
#using either 2 x 2 or 3 x 3  cells_per_block  obtains reasonable accuracy in most cases.
 
#AS PER MY CALCULATION, THE NUMBER OF POSSIBLE BLOCKS =
#      (Number of rows after dividing the image into cells + 1 - Number of rows per block ) x (Number of columns after dividing the image into cells + 1 - Number of columns per block )
#where Number of rows per block = Number of cells in block when counted from top to bottom
#      Number of columns per block = Number of cells in block when counted from left to right
 
#Finally, each pixel contributes a weighted vote to the histogram — the weight of the vote is simply the gradient magnitude |G| at the given pixel.
#
 
#########################################
#Step-5: Concate final feature vectors ##
#########################################
#Finally, after all blocks are normalized, we take the resulting histograms, concatenate them, and treat them as our final feature vector.
 
 
 
from skimage import exposure
from skimage import feature
import cv2

#image = cv2.imread("../images/florida_trip.png")
image = cv2.imread("../images/trex.png")
#Assume that we are using the following:
#Pixels per cell = 8 x 8
#Cells per block = 2 x 2
#Orientations = 9 (this means each cell will give us 9 features)
 
#trex.png image size is 228 x 350 pixels
#pixels per cell = 8 x 8
#therefore 28 x 43 cells = 1204 cells
#features per cell = 9.
#cells per block = 2 x 2
#Total blocks = (28 + 1 -2) * (43+1 -2) = 27 x 42 = 1134 blocks.
#Each block will have 9 + 9 + 9 + 9 = 9*4 = 36 features (since each block has 4 cells and each cell will have 9 features)
#Hence total features obtained = 1134 * 36 = 40824
 
#The above calculation demonstrates that the input image size also dictates the number of HOG features
 
print(image.shape)
 
#If we use visualize=True, then we will get 2 return values: Histogram features and histogram image itself.
(H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True,block_norm="L1",visualize=True)
 
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

c2.imshow("HOG Image", hogImage)

cv2.waitKey(0)

print(hogImage.shape)

print(H.shape)
 
#If we do not use visualize=True, then we will get 2 return values. Histogram features and histogram image itself. Default is visualize=False.
H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True,block_norm="L1")
print(H.shape)
 
#See the material to see the case study of extracting HOG features by contouring the object in an image.
#In the case study (detecting the Car logos), Adrain had to extract the HOG after applying contouring, to avoid noise
#caused by the image parts, which do not belong to the actual logo.
 
#Suggestions (as per Adrian)
#HOG descriptors are very powerful;
#however, it can be tedious to choose the correct parameters for the number of orientations , pixels_per_cell ,
#and cells_per_block , especially when you start working with object classification.
#As a starting point, I tend to use orientations=9 , pixels_per_cell=(4, 4) , and cells_per_block=(2, 2) ,
#then go from there.
#It’s unlikely that your first set of parameters will yield the best performance; however,
#it’s important to start somewhere and obtain a baseline — results can be improved via parameter tuning.

 
#HOG Pros and Cons
#Pros:
 
#Very powerful descriptor.
#Excellent at representing local appearance.
#Extremely useful for representing structural objects that do not demonstrate substantial variation in form (i.e. buildings, people walking the street, bicycles leaning against a wall).
#Very accurate for object classification.
 
#Cons:
#Can result in very large feature vectors, leading to large storage costs and computationally expensive feature vector comparisons.
#Often non-trivial to tune the orientations , pixels_per_cell , and cells_per_block  parameters.
#Not the slowest descriptor to compute, but also nowhere near the fastest.
#If the object to be described exhibits substantial structural variation (i.e. the rotation/orientation of the object is consistently different), then the standard vanilla implementation of HOG will not perform well.
 
#Questions:
 
#1. HOG descriptors operate on:
#A. The thresholded, binary representation of an image.
#B. Gradient magnitude representation of an image.
#C. The Canny edge map of an image.
#Ans: B
 
#2. All are forms of image normalization/transformation except:
#A. Gamma/power law.
#B. Square-root.
#C. Variance.
#D. Chi-squared.
#Ans: D 
  
#3. Given an input image of 256 x 256 pixels and pixels_per_cell=(16, 16), we would have a total of how many cells in our image?
#A. 128
#B. 64
#C. 512
#D. 256
#Ans: D
 
#4. It’s important to consider the dimensions (i.e. width and height) of the input image prior to computing HOG features:
#  True
 
 
 
