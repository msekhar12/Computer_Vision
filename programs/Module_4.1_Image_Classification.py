#4.1.1 Image classification
#Image classification, at the very core, is the task of assigning a label to an image from a pre-defined set of categories.
 
#Practically, this means that given an input image, our task is to analyze the image and return a label that categorizes the image.
#This label is (almost always) from a pre-defined set. It is very rare that we see “open-ended” classification problems where the list of labels is infinite.
 
#More formally, given our input image of W x H pixels, with 3 channels, Red, Green, and Blue, respectively,
#our goal is to take the W x H x 3 = N pixels and figure out how to accurately classify the contents of the image.
 
#In Images see the image image_classification_beach-865x550.jpg


import cv2
import numpy as np
 
image = cv2.imread("./images/image_classification_beach-865x550.jpg")
 
cv2.imshow("beach", image)
cv2.waitKey(0)
 
#In the displayed image (image_classification_beach-865x550.jpg), we might describe the image as follows:
 
#Spatial: The sky is at the top of the image and the sand/ocean are at the bottom.
#Color: The sky is dark blue, the ocean water is lighter than the sky, while the sand is tan.
#Texture: The sky has a relatively uniform pattern, while the sand is very coarse.
 
#Using "image descriptors" and "deep learning" methods, we can encode the above description of the image, and extract and quantify regions of an image.
#Some descriptors are used to encode spatial information. Others quantify the color of an image. And other features are used to characterize texture. The list goes on from there.
#Finally, based on these characterizations of the image, we can apply machine learning to “teach” our computers what each type of image “looks like.”
 
#But it is not that simple. Because once we get to examining images in the real world, we are faced with many, many challenges.
#Some of the Challenges are given below:
#1. Viewpoint Variation: The object can be rotated/oriented in multiple dimensions with respect to how the object is photographed and captured
#2. Scale Variation: The size of an object in the image will change dramatically depending on how far the camara is from the object or how much zoom we use while
#                    capturing the picture.
#3. Deformations: Objects are subject to deformation in unpredictible shaped in the image. For example a giraffe's neck might look shorter when it tries to eat
#                 something lying on the ground, while the giraffe's neck might look stretched when it tries to reach the top of a tree to grab food.
#4. Occlusions: Our image classification system should also be able to handle occlusions, where large parts of the object we want to classify are hidden from view in the image.
#5. Illumination: Based on the lighting conditions the image could look dramatically different
#6. Background clutter: We are only interested in one particular object or specific objects in an image; however, there might be lot of background clutter in the image,
#and this makes the computer vision algorithm very difficult to extract the details of specific object in the image. For example, imagine that we are developing a CV application
#that identifies a specific person standing in a huge crowd. Due to the presence of many people it becomes very difficult for the CV algorithm to identify the presence of the
#target person in the image
#7. Intra-class variation: Our target object might look in different forms. For example a CV application that identifies a chair in the image must account for various forms of chairs.
#From comfy chairs that we use to curl up and read a book, to chairs that line our kitchen table for family gatherings, to ultra-modern art deco chairs found in
#prestigious homes, a chair is still a chair — and our image classification algorithms must be able to categorize all these variations correctly.
 
#In general, we try to frame the problem as best we can. We make assumptions regarding the contents of our images and
#which variations we want to be tolerant to. And we consider the scope of our project — what is the end goal? And what are we trying to build?
 
#Successful computer vision and image classification systems deployed to the real world make careful assumptions and
#considerations before a single line of code is ever written. If you take too broad of an approach, such as
#“I want to classify and detect every single object in my bathroom,” then your classification system is unlikely to perform
#well unless you have years experience in building image classifiers — and even then, there is no guarantee to the success of the project.
 
#But if you frame your problem and make it more narrow in scope, such as “I want to recognize brands of toothpaste on the bathroom counter,”
#then your system is much more likely to be accurate and functioning.
 
#The key takeaway here is to always consider the scope of your image classifier — especially when you are first getting started.
#Keep the scope as tight and well-defined as possible, and you’re much more likely to end up with a working system.
 
#Q. At the very core, image classification is:
#A. The task of assigning a label to an image from a predefined set of categories.
 
#Q. The semantic gap is:
#The difference in how a human perceives the contents of an image and how an image is represented in a way a computer can process.
 
#Q. All of the following are challenges in image classification except:
#A. Scale variation.
#B. Deformation.
#C. External class variation.
#D. Viewpoint variation.
#Answer: C
