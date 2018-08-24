#10.3: Color histograms
 
#Color channel statistics quantify an image by using the statistics of the pixel intensities in each of the color channel of the image.
#Color histogram represents the frequency of the pixel intensities. A pixel can have a value between 0 to 255 as its intensity.
#And color histogram just counts how many pixels are existing in each of the possible pixel values (0 to 255). We can also bucket the pixel intensities
#(like [0, 8), [8, 16), [16, 24),.....,[248, 256) ) and count the number of pixels in the image present in each of the bucket.
 
#For colored images we can also compute 2D and 3D histograms.
#2D histogram is a kind of correlation matrix. It counts the number of pixels existing at each of the pixel intensities for 2 color channels.
#In 2D we will have 256 X 256 combinations of pixel intensities (assuming that we did not bucket the pixel intensities)
#We will get 3 possible 2D histograms: The first one for R-G, the second one for G-B and the third one for R-B
 
#The 3D histogram will consider all the 3 color channels and obtain the histogram.
#Given that we will get a huge number of possible pixel intensity combinations, we usually select a bucket size of 8 or 3 for 2D and 3D histograms.
 
#Example statement to compute a 2D histogram. See histogram notes in module-1 for more details.
# plot a 2D color histogram for green and red
# hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],  [0, 256, 0, 256])
 
#Advantages of using Color histograms and image statistics:
#1. The major advantage of using these image descriptors (Color histograms and image statistics) is independence of image size.
#Irrespective of the image size we always get the same feature vector size.
#However it is important to normalize the Color histogram statistics (divide the pixel intensity counts by the number of pixels in the image)
 
#To execute use the following command:
#python Module_10_Lesson_3_Color_histograms.py -d ../images/dataset/ -k 2 
    
# import the necessary packages
import cv2
import imutils
 
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2

#Create a class that helps us to get 3D histograms for a colored image
#It is optimal to define Image descriptors as classes, because you rarely ever extract features from a single image alone.
#You instead extract features from an entire dataset of images. Furthermore, you expect that the features extracted from all
#images utilize the same parameters — in this case, the number of bins for the histogram.
#It wouldn’t make much sense to extract a histogram using 32 bins from one image and then 128 bins for another image if you
#intend on comparing them for similarity or clustering them into groups of similar images.
 
#It is very important that we perform normalization step (cv2.normalize()).
#If we did not, then images with the exact same contents but different sizes would have dramatically different histograms.
#Instead, by normalizing our histogram we ensure that the width and height of our input image has no effect on the output histogram.
 
#This is available in the pyimagesearch package
class LabHistogram:
                def __init__(self, bins):
                                # store the number of bins for the histogram
                                self.bins = bins

                def describe(self, image, mask=None):
                                # convert the image to the L*a*b* color space, compute a histogram,
                                # and normalize it
                                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                                hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins,
                                                [0, 256, 0, 256, 0, 256])

                                # handle if we are calculating the histogram for OpenCV 2.4
                                if imutils.is_cv2():
                                                hist = cv2.normalize(hist).flatten()

                                # otherwise, we are creating the histogram for OpenCV 3+
                                else:
                                                hist = cv2.normalize(hist,hist).flatten()

                                # return the histogram
                                return hist
 
#We will create 3D histogram statistics for all images, and apply K-Means clustering to cluster the images into 2 groups.


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the input dataset directory")
ap.add_argument("-k", "--clusters", type=int, default=2,
                help="# of clusters to generate")
args = vars(ap.parse_args())

# initialize the image descriptor along with the image matrix
# Each image will have 512 features
desc = LabHistogram([8, 8, 8])
data = []

# grab the image paths from the dataset directory
imagePaths = list(paths.list_images(args["dataset"]))
#print(imagePaths)
 
#Image paths will be a list of file paths sorted by name
imagePaths = np.array(sorted(imagePaths))
#print(imagePaths)
 
#Note that imagePaths is a numpy array
 
# loop over the input dataset of images
for imagePath in imagePaths:
                # load the image, describe the image, then update the list of data
                image = cv2.imread(imagePath)
                hist = desc.describe(image)
                data.append(hist)
#data will be a 512 sized feature vector for each image.
#print(data)   
 
#Since we have 10 images in the dataset, we will have 10 feature vectors and each feature vector will be of size 8
#As the data in the imagePaths is sorted, we can associate the feature vector to the corresponding image in imagePaths list
#print(len(data))
 
# cluster the color histograms
clt = KMeans(n_clusters=args["clusters"])
labels = clt.fit_predict(data) 
 
#Labels will contain either 0 or 1 (one label for each cluster, as we will supply the desired number of clusters as 2)
#print(labels)
 

# loop over the unique labels
for label in np.unique(labels):
                # grab all image paths that are assigned to the current label
    # since imagePaths is a numpy array, we can apply np.where
                labelPaths = imagePaths[np.where(labels == label)]

                # loop over the image paths that belong to the current label
                for (i, path) in enumerate(labelPaths):
                                # load the image and display it
                                image = cv2.imread(path)
                                cv2.imshow("Cluster {}, Image #{}".format(label + 1, i + 1), image)

                # wait for a keypress and then close all open windows
                cv2.waitKey(0)
                cv2.destroyAllWindows()
   
#Questions:
#Color histograms can be used to quantify the contents of an image.
#True