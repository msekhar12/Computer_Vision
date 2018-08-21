#10.1 Image descriptors

#There are three important terms related to the quantifying an image:
#1. Feature Vector
#2. Image Descriptor
#3. Feature descriptor

#Feature vector: Simply put, a feature vector is a list of numbers used to represent an image.
#Given an NXM pixel image, the "image descriptor" can output a d-dimensional vector.
#For example if your input is a 128X128 pixel image, then the image descriptor will process this 
#128X128 image into a 512 dimensions vector (or simply a vector of length 512)
#Simply put: a feature vector is nothing more than a list of numbers used to represent and quantify an image.

#Of course, these feature vectors cannot be arbitrary and filled with random, meaningless values! 
#We need to define methodologies to examine our images and extract meaningful feature vectors from them.

#These algorithms and methodologies used to extract feature vectors are called "image descriptors" and "feature descriptors".

#Image descriptor: An image descriptor is an algorithm and methodology that governs how an input image is "globally" quantified 
#and returns a feature vector abstractly representing the image contents.
#The key term to understand here is "global" — this implies that we are examining the entire image and 
#using the whole image in the computation of our feature vector (not just 50% or 99% of the image. We will use 100% of the image)

#|---------------|         |-----------------|          ___________________
#|COMPLETE IMAGE |  ====>  |IMAGE DESCRIPTOR |  =====> |FEATURE VECTOR 1XD |
#|   (NXM) Image |         |-----------------|         |___________________| 
#|---------------|  


#One of the primary benefits of image descriptors is that they tend to be much simpler than "feature descriptors". 
#Furthermore, once extracted, the feature vectors derived from image descriptors can be immediately passed down the pipeline 
#to other computer vision methods, such as creating a classifier to recognize the contents of an image or building an image search engine.

#However, this simplicity often comes at a price. 
#Often times, while basic and simple to use, our image descriptors are not robust to changes in how the 
#image is rotated, translated, or how viewpoints of an image change. 
#If that is the case, we’ll often times need to use the more powerful "feature descriptors".

#But if we wanted to describe multiple regions in our image separately. 
#For example in our image we want to separately describe the sky, faces, and trees, then IMAGE DESCRIPTOR cannot be used
#as IMAGE DESCRIPTOR will create a gereric feature vector for the whole image. The FEATURE DESCRIPTOR can help use 
#to locally quantify various areas of interest in an image.

#Feature Descriptor:A feature descriptor is an algorithm and methodology that governs how an input region of an image 
#is locally quantified. A feature descriptor accepts a single input image and returns multiple feature vectors.

#                                                           ___________________
#                                              |=========> |FEATURE VECTOR 1XD |
#                                              |           |-------------------| 
#|---------------|         |-------------------|           ___________________
#|COMPLETE IMAGE |  ====>  |FEATURE DESCRIPTOR |=========> |FEATURE VECTOR 1XD |
#|   (NXM) Image |         |-------------------|           |___________________| 
#|---------------|                             |            ___________________
#                                              |=========> |FEATURE VECTOR 1XD |
#                                                          |-------------------|  

#Image descriptor: 1 image in, 1 feature vector out.
#Feature descriptor: 1 image in, many feature vectors out.

#Feature descriptors tend to be much more powerful than our basic image descriptors since they take into account the locality of regions in an 
#image and describe them in separately. Feature descriptors also tend to be much more robust to changes in the input image, such as rotation, 
#translation, orientation (i.e. rotation), and changes in viewpoint.

#However, this robustness and ability to describe multiple regions of an image comes at a price. In most cases the feature vectors extracted 
#using feature descriptors are not directly applicable to building an image search engine or constructing an image classifier in their current state.

#This is because each image is now represented by "multiple feature vectors" rather than just "one".

#To remedy this problem, we construct a bag-of-visual-words, which takes all the feature vectors of an 
#image and constructs a histogram, counting the number of times similar feature vectors occur in an image.

#Questions:
#1. A feature vector is NOT:
#A. A method/algorithm applied to an image to quantify it.
#B. A list of numbers used to quantify of an image/region of an image.
#C. An abstract representation of the contents of an image.
#Answer: A


#2. What is the primary difference between image descriptors and feature descriptors?
#Answer: Image descriptors are used to globally quantify an image, while feature descriptors are used to locally quantify an image.

