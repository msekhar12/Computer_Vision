#from __future__ import print_function
#Read notes from my book, and refer to gurus notes if needed.
#This code does not use spatial encoding step.
 
# import the necessary packages
from skimage import feature
import numpy as np

lass LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation of the image, and then
        # use the LBP representation to build the histogram of patterns
           
        #method can be one of the following (http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern):
        #'default': original local binary pattern which is gray scale but not rotation invariant.
        #'ror': extension of default implementation which is gray scale and rotation invariant.
        #'uniform': improved rotation invariance with uniform patterns and finer quantization of
        #the angular space which is gray scale and rotation invariant.
        #'nri_uniform': non rotation-invariant uniform patterns variant which is only gray scale invariant
        #'var': rotation invariant variance measures of the contrast of local image texture which is rotation
        #but not gray scale invariant.
            
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        #As per my understanding, addition of a really low value 1e-7 will avoid the indefinite form of
        #division (division by 0)
       
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist
 
 
 
# import the necessary packages
#from __future__ import print_function
#from pyimagesearch import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the dataset of shirt images")
ap.add_argument("-q", "--query", required=True, help="path to the query image")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor and initialize the index dictionary
# where the image filename is the key and the features are the value

desc = LocalBinaryPatterns(24, 8)
index = {}

# loop over the shirt images
for imagePath in paths.list_images(args["dataset"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    # update the index dictionary
    #If windows use "\\" else use "/"
    filename = imagePath[imagePath.rfind("\\") + 1:]
    index[filename] = hist       

# load the query image and extract Local Binary Patterns from it
query = cv2.imread(args["query"])
queryFeatures = desc.describe(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

# show the query image and initialize the results dictionary
cv2.imshow("Query", query)
cv2.waitKey(0)
results = {}

# loop over the index
for (k, features) in index.items():
                # compute the chi-squared distance between the current features and the query
                # features, then update the dictionary of results
                d = 0.5 * np.sum(((features - queryFeatures) ** 2) / (features + queryFeatures + 1e-10))
                results[k] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()])[:3]

# loop over the results
for (i, (score, filename)) in enumerate(results):
    #print(filename)
    #print(args["dataset"] + "/" + filename)
    # show the result image
    print("#%d. %s: %.4f" % (i + 1, filename, score))
    image = cv2.imread(args["dataset"] + "/" + filename)
    #print(image.shape)
    cv2.imshow("Result #{}".format(i + 1), image)
    cv2.waitKey(0)   
 
#Suggestions when using Local Binary Patterns
#The main point to realize when utilizing local binary patterns is that the radius and
#number of points has a dramatic effect on
#(1) the dimensionality of your feature vector and
#(2) computational efficiency — provided you are not using the rotation invariant uniform
#implementation of LBPs, in which case your feature vector size will remain fixed at 25-d,
#but your computation times can increase.
 
#Furthermore, the larger your radius and number of points to consider are,
#the slower your extraction will be. At points, this extraction becomes prohibitively slow, so take care when using LBPs.
 
#Personally, I(Adrain) always start off with p=8 and r=1.0 and perhaps work my way up to p=24 and r=3.0,
#increasing the radius further to see if my accuracy improves.
#I(Adrain) also tend to use rotation invariant LBPs whenever possible as they have substantially
#smaller feature vector size and are easier to compute a histogram for.       
 
 
 
#Questions:
#1. Local Binary patterns are used to characterize and quantify the (fill in the blank) of an image:
#A. Texture/pattern
#B. Shape
#C. All three: texture, shape, and color
#D. Color
#Answer: A