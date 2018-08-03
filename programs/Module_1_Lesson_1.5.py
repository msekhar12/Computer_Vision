#Module:1
#Chapter: 1.5 Kernels

#If we think of an image as a big matrix, then we can think of a kernel or convolutional matrix
#as a tiny matrix that is used for blurring, sharpening, edge detection, and other image processing functions.

#Essentially, this tiny kernel sits on top of the big image and slides from left to right and
#up to down, applying a mathematical operation at each (x, y)-coordinate in the original image.
#Again, by applying kernels to images we are able to blur and sharpen them

#We can also use convolution to extract features from images
#and build very powerful deep learning systems (More on this in Module-8, Convolutional Neural Nets)

#We will define a kernel as a matrix, and move that matrix over the image matrix. The image's pixel
#located at the center of the kernel will be replaced by multiplying and summing the kernel elements
#(including the kernel's center element) with the image elements

#Kernels can be an arbitrary size of M X N pixels, provided that both M and N are odd integers
#For a 3 X 3 kernel, the center is located at [1,1].
#For a 5 X 5 kernel, the center is located at [3,3].
#For a 9 X 9 kernel, the center is located at [5,5].
#For a 5 X 9 kernel, the center is located at [3,5].

#We cannot use even dimensions to define kernels, as we cannot identify the center pixel
#for even dimensions kernel. OpenCV will trow an error if we try to use even dimensional kernel

###############
#Convolution ##
###############

#In image processing, convolution requires three components:
#1. An input image.
#2. A kernel matrix that we are going to apply to the input image.
#3. An output image to store the output of the input image convolved with the kernel.

#Convolution itself is very easy. All we need to do is:
#1. Select an (x, y)-coordinate from the original image.
#2. Place the center of the kernel at this (x, y) coordinate.
#3. Multiply each kernel value by the corresponding input image pixel value,
#   and then take the sum of all multiplication operations.
#Example:
#Kernel = [[-1, 0,  1],
#          [-2, 0,  2],
#          [-1, 0,  1]]
#
#Image = [[93, 139, 101],
#         [26, 252, 196],
#         [135, 230, 18]]
 
#Convolution = Element wise multiplication of Kernel and Image (NOT matrix multilication), and taking the sum
# [[-93, 0, 101],
#  [-52, 0, 392],
#  [-135, 0 18]]
# = 231

#Hence, Convolution is simply the sum of element-wise matrix multiplication between the kernel and the neighborhood
#that the kernel covers of the input image.

#But how are the edge pixels in the image are handled?
#They may be just made black (see http://setosa.io/ev/image-kernels/)
#As per the website http://setosa.io/ev/image-kernels/, we can use the following matrices as kernels:
#blur:
#[[0.0625, 0.125, 0.0625],
# [0.125,  0.25 , 0.125],
# [0.0625, 0.125, 0.0625]]

#bottom sobel:
#[[-1, -2, -1],
# [0, 0, 0],
# [1, 2, 1]]

#emboss:
#[[-2, -1, 0],
# [-1, 1, 1],
# [0, 1, 2]]

#identity
# [0, 0, 0],
# [0, 1, 0],
# [0, 0, 0]]

#left sobel:
#[[1,0,-1],
# [2,0,-2],
# [1,0,-1]]
#

#outline:
#[[-1,-1,-1],
# [-1,8,-1],
# [-1,-1,-1]]

#
#right sobel
#[[-1,0,1],
# [-2,0,2]

# [-1,0,1]]
#
#sharpen
#[[0,-1,0],
# [-1,5,-1],
# [0,-1,0]

#
#top sobel
#[[1,2,1],
# [0,0,0],
# [-1,-2,-1]
#

 

 