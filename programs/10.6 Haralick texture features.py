#10.6 Haralick texture features
#See my notebook for description of Haralick features.
 
 
#The following statements show how to get haralick features.
#import mahotas
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#features = mahotas.features.haralick(gray).mean(axis=0)
 
 
# import the necessary packages
from sklearn.svm import LinearSVC
import argparse
import mahotas
import glob
import cv2
import pandas as pd
 
#Read the image:
image = cv2.imread("./images/tablecloth_01.png")
 
#display it
cv2.imshow("image", image)
cv2.waitKey(0)
 
#convert to gray
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
#display it
cv2.imshow("image", image)
cv2.waitKey(0)
 
# extract Haralick texture features in 4 directions, then take the
features = mahotas.features.haralick(image)
print(features)
 
#You will get 13 features in each direction.
#left to right
#top to bottom
#top left to bottom right
#top right to bottom left
 
 
#Convert to pandas data frame and display how the shape looks.
#we should see 4 rows and 13 columns
print(pd.DataFrame(features))
 
# extract Haralick texture features in 4 directions, then take the
# mean of each direction (mean by row for each column, i.e, for each column compute the mean using the row values)

features = mahotas.features.haralick(image).mean(axis=0)

print(features)
 
#For a complete description see pyimagesearchgurus
 
#As per Adrain (pyimagesearchgurus):
#Suggestions when using Haralick texture:
#The most important suggestion when using Haralick texture features is to take the average of each of the dimensions.
#This will improve the accuracy of your descriptor and make it slightly more robust to rotation.
 
#Secondly, remember that Haralick texture features are extracted from the entire image.
#If you only want to describe part of an image, you’ll need to extract the ROI first, and then extract Haralick features from the ROI.
 
#Lastly, be sure to pay attention to the ignore_zeros  parameter of the mahotas.features.haralick  function.
#This parameter controls whether or not values of 0 should be included in the construction of the GLCM (and thus the Haralick descriptor).
#In general, you’ll want to set ignore_zeros=False  since you want to include black pixels in your calculation.
#However, if you have applied background subtraction or some other masking technique, then these pixels are likely black —
#and thus you’ll want to set ignore_zeros=True  so they are not included in the calculation.
 
#Haralick Pros and Cons
#Pros:
 
#Very fast to compute.
#Low dimensional — requires less space to store the feature vector, and facilitates faster feature vector comparisons.
#No parameters to tune.
#Cons:
#Not very robust against changes in rotation.
#Very sensitive to noise — small perturbations in the grayscale image can dramatically affect the construction of the GLCM, and thus the overall Haralick feature vector.
#Similar to Hu moments, basic statistics are often not discriminative enough to distinguish between many different kinds of textures.

#Q. Computing a Haralick texture feature vector requires that we first compute the GLCM:
#True
 
#Q. We take the average of the 4 GLCM directions to:
#A.  Make the feature vector more robust to changes in translation.
#B.  Reduce the size of the feature vector.
#C.  Make the feature vector more robust to changes in rotation.
#D.  Make the feature vector more robust to changes in scaling.
#Answer: C
 
#Q. Compute Haralick features of sand.jpg
 
image = cv2.imread("./images/sand.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
features = mahotas.features.haralick(image).mean(axis=0)
print(features)
 
#[9.34e-05, 8.62e+02, 7.09e-01, 1.48e+03, 5.59e-02, 2.41e+02, 5.07e+03, 8.17e+00, 1.37e+01, 7.73e-05, 5.86e+00, -1.04e-01, 8.80e-01]