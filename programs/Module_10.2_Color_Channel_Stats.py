#10.2 Color Channel statistics

#As a quick recap, feature vectors are a list of numbers used quantify and abstractly quantify images. 
#These feature vectors can then be compared using distance functions/similarity metrics to determine how “similar” two images are. 
#Feature vectors are also used as inputs to machine learning models that can be trained to recognize the contents of an image.

#Color Channel Statistics
#By computing basic statistics, such as mean and standard deviation for each channel of an image, 
#we are able to quantify and represent the color distribution of an image. Therefore, if two images 
#have similar means and standard deviations, we can assume that these images have similar color distributions

#We will use statistics like mean, standard deviation, skewness, kurtosis of pixel intensity distribution in each of the color channels.

#For uni-variate distribution skewness is defined as sum((x - mean)^3/std.dev^3)/n
#Skewness gives a quantative measurement of skewness of the data as compared to the normal distribution.
#
#Kurtosis gives a quantative measurement of how bulky are the tails of the distribution compared to the normal distribution.

#The kurtosis of std normal distribution is 3.

#Kurtosis of a uni-variate distribution is sum((x - mean)^4/std.dev^4)/n

#Note that in computing the kurtosis, the standard deviation is computed using n in the denominator rather than n - 1

#Some definitions of kurtosis use the following. This is called excess kurtosis, and finds how much is the variable's kurtosis as compared to the std. normal kurtosis:
#kurtosis = sum((x - mean)^4/std.dev^4)/n - 3 



#The color channel image descriptor can be broken down into three steps:

#Step 1: Separate the input image into its respective channels. For an RGB image, we want to examine each of the Red, Green, and Blue channels independently.
#Step 2: Compute various statistics for each channel, such as mean, standard deviation, skew, and kurtosis.
#Step 3: Concatenate the statistics together to form a “list” of statistics for each color channel — this becomes our feature vector.



#Note: Comparing images based on their color distributions is almost as old as the computer vision field itself. 
#When used in practice, we assume that images that have similar color distributions also have similar visual contents. 
#This is clearly not always possible, as a white horse in a green field will have a similar color distribution as a white monkey in the rainforest. 
#However, it’s a nice assumption to make when using small datasets and can lead to decent results.

#Color statistics are often not as powerful as histograms, but they are also more compact. 

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import cv2


# grab the list of image paths and initialize the index to store the image filename
# and feature vector
imagePaths = sorted(list(paths.list_images("./images/dinos/")))
index = {}

#Download the trex_01.png, trex_02.png, trex_03.png and trex_04.png images to ./images/dinos
 
# loop over the image paths
for imagePath in imagePaths:
    # load the image and extract the filename
    image = cv2.imread(imagePath)
    
    #Just use the file names like trex01.png etc as the keys in the index dictionary
    filename = imagePath[imagePath.rfind("/") + 1:]

    # extract the mean and standard deviation from each channel of the
    # BGR image, then update the index with the feature vector
    (means, stds) = cv2.meanStdDev(image)
    print("Means for image {}:".format(filename))
    print(means)
    print("std dev for image {}:".format(filename))
    print(stds)
    features = np.concatenate([means, stds]).flatten()
    print("flattened list: {}".format(features))
    index[filename] = features

#How does the flatten() works?
#Example
import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])


print(np.concatenate([a,b]))
#will print:
#array([[1, 2],
#       [3, 4],
#       [5, 6],
#       [7, 8]])

print(np.concatenate([a,b]).flatten())
#will print:
#array([1, 2, 3, 4, 5, 6, 7, 8])

#Back to the topic....
#Let us print the index dictionary:
for k, v in index.items():
    print(k, v)

#Eucleadian Distance    
#If (x1, y1) and (x2, y2) are vectors in a 2D space, then the eucleadian distance between them is given by:
#sqrt((x1-x2)^2 + (y1 - y2)^2)

#If two vectors are similar the distance must be less and if they are dissimilar then it should be more.

#We will search the images which are neares to the trex01.png image

# display the query image and grab the sorted keys of the index dictionary
query = cv2.imread(imagePaths[0])
cv2.imshow("Query (trex_01.png)", query)
keys = sorted(index.keys())
 
# loop over the filenames in the dictionary
for (i, k) in enumerate(keys):
	# if this is the first image, ignore it
	if k == "trex_01.png":
		continue
 
	# load the current image and compute the Euclidean distance between the
	# query image (i.e. the 1st image) and the current image
	image = cv2.imread(imagePaths[i])
	d = dist.euclidean(index["trex_01.png"], index[k])
 
	# display the distance between the query image and the current image
	cv2.putText(image, "%.2f" % (d), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow(k, image)
 
# wait for a keypress
cv2.waitKey(0)



#Questions:
#1. When comparing feature vectors we assume:
#   A.  We don't use distance metrics/similarity functions like the Euclidean distance to compare feature vectors.
#   B.  The larger the Euclidean distance is, the more similar images are.
#   C.  The smaller the Euclidean distance, the more similar images are.
#Answer:  C

#Find the mean and std dev of the color channels for the image raptors_01.png

image = cv2.imread("./images/raptors_01.png")
cv2.imshow("image", image)

cv2.waitKey(0)

(means_1, stds_1) = cv2.meanStdDev(image)
print(np.concatenate([means_1, stds_1]).flatten())
feature_vec_1 = np.concatenate([means_1, stds_1]).flatten()
image = cv2.imread("./images/raptors_02.png")
cv2.imshow("image", image)

cv2.waitKey(0)

(means_2, stds_2) = cv2.meanStdDev(image)
print(np.concatenate([means_2, stds_2]).flatten())
feature_vec_2 = np.concatenate([means_2, stds_2]).flatten()

d = dist.euclidean(feature_vec_1, feature_vec_2)
print(d)