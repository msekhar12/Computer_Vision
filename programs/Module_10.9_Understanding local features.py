#10.9: Understanding local features
#Image that we are developing a book search mobile app, which helps user to snap a picture of a book cover
#via his mobile phone and it searches a set of book cover images in a database and renders the closely matched images.
 
#If we use global image descriptors like Haralick texture, HOG or LBP, then the features obtained will be at the global
#level, and there will be unnecessary noise in the image. For example if the user holds the book in his hand and snaps the
#photo, then his hand will be included in the image and the global image descriptors will quantify the whold image including the

#image part where his hand is present. The noise in the global features could drastically reduce the accuracy of
#our Predictive models.
 
#The solution is to instead use local features, where we describe only small, local regions of the image that are deemed “interesting”
#instead of the entire image. These regions should be unique, easily compared, and ideally carry some sort of semantic meaning
#in relation to the contents of the image.
 
#One of the biggest differences between image descriptors and our local feature descriptor counterparts is that we’ll end up
#generating multiple feature vectors per image  — one for each “interesting” region
 
#Local feature algorithms in computer vision attempt to find interesting, repeatably unique regions in an image — followed by quantifying
#the region surrounding each of these patches.
 
#The process of finding and describing interesting regions of an image is broken down into two phases:
#1. keypoint detection and
#2. feature extraction
 
#Keypoint detection algorithms will find interesting regions in the image. These regions could be edges, corners, “blobs”,
#or regions of an image where the pixel intensities are approximately uniform.
#Irrespective of the keypoint detection algorithm we use the interesting regions identified in the image are called as "keypoints".
#At the very core, keypoints are simply the (x, y)-coordinates of the interesting, salient regions of an image.
 
#Then for each of our keypoints, we must describe and quantify the region of the image surrounding the keypoint by extracting a feature vector.
#This process of extracting multiple feature vectors, one for each keypoint, is called feature extraction.
 
#Challenges of keypoint detection and local features:
#1. repeatability: Given two images containing the same object, we want to be able to detect the same keypoints in both images,
#                  even if the second image contains dramatically different viewpoint angles or lighting changes.
#2. quality: a high-quality keypoint will be both repeatable and contain enough information to be
#            discriminative amongst the other regions of an image
#3. speed: some algorithms (for keypoint detection and local feature extract) are much faster than others but tend to sacrifice accuracy.
#Others are much slower, but the speed trade-off leads to higher accuracy and quality keypoints and descriptions.
#4. invariant: we would like each of the feature vectors obtained from the regions surrounding each keypoint to be
#invariant to rotation, scale, lighting changes, contrast, and pretty much all the challenges we specified in the
#What is image Classification module.
 
#Local features operate in two phases.
#First, we must find the interesting and salient regions of an image called “keypoints” — this process is called keypoint detection.
#Then, for each of our keypoints, we must quantify and extract a feature used to describe this local region of the image.

#The process of extracting a feature vector for each of the local regions of an image is called feature extraction.
 
#Questions:
#1. Local features are appropriate when:
#A. When we want to extract multiple feature vectors from an image.
#B. We are only interested in characterizing local, salient regions of an image — not the entire global image.
#C. When are are interested in describing the entire global image — not just small regions of it.
#Ans: B
 
#2. A “feature” is a region of an image that is:
#a. Common, hard to distinguish between.
#b. Unique and easily recognizable.
#c. An area of low texture and shape.
#d. A "flat" region of an image.
#Ans: b
 
#3. Corners are considered to be good features because:
#A. Shifting the region to the left or right or up and down would make the patch look substantially different.
#B. Corner regions are computationally simple to find in an image.
#C. Corner regions are easy to detect.
#Ans: A
 
  
#4. When detecting keypoints and extracting features we end up with:
#A. Multiple feature vectors, one for each keypoint.
#B. One feature vector to characterize the entire image.
#C. Multiple feature vectors, one for each pixel in the image. 
#Ans: A