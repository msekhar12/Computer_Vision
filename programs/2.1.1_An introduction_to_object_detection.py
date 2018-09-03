#2.1.1: An introduction to object detection
#Image classification deals with identifying whether a spaecific object is present in the image or not.
#For example we can train an image classifier to identify whether a stop sign is present in the image.
#While it some what easier to train an image classifier, detecting object is more challenging.
#Object detection deals with not only identifying whether an object is present in the image,
#it also determines the location of the object in the image.
 
#Object recognition is hard for the same reason image classification is hard.
#Objects in the real-world can exhibit substantial variations in viewpoint, scale, deformation, occlusion, illumination,
#background clutter, and intra-class variation (see https://github.com/msekhar12/Computer_Vision/blob/master/programs/Module_4.1_Image_Classification.py)
 
#Object detection has many applications in real world, some of which are given below:
#Detecting faces in an image/video (for better focus)
#Detecting cars, sign boards etc in an image/video (for self-driving cars)
#Detecting presence of a person or people in security systems
 
#In order to determine the exact location of an object in an image, we need to extend our image classification
#knowledge and construct an object detector. As the name suggests, an object detector is used to scan an
#image and look for the presence of a given object (i.e. a stop sign, motorcycle, dog, etc.) and identify its location.
 
#Object detectors tend to be substantially more challenging to build than simple image classifiers;
#however, the results we obtain from our object detectors are often times much more useful.
 
#Q. Object detection can be seen as a specialized form of image classification.
#True
 
#Q. The goal of object detection is to report the location and size of a given object in an image.
#True
 
#Q. The primary difference between object detection and image classification is:
#a. Image classification seeks to find specific objects in a portion of the image, while object detection attempts to classify the entire contents of an image.
#b. Image classification attempts to classify the entire contents of an image, while object detection seeks to localize specific objects in part of an image.
#c. There is no difference between object detection and image classification.
#Answer: b