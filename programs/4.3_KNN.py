#4.3: k-Nearest Neighbor classification

#Simply put, the k-NN algorithm classifies unknown data points by finding the most common class among the k closest examples. 

#In order to apply the k-Nearest Neighbor classifier, we first need to select a distance metric or a similarity function. 

#Eucledian distance (L2 Norm) between two vectors (x1, y1, z1) and (x2, y2, z2) is defined as:
#sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)

#Manhattan distance is defined as:
#mod(x1 - x2) + mod(y1 - y2) + mod(z1 - z2)

# In reality, you can use whichever distance metric/similarity function most suits your data (and gives you the best classification results)
# However, for the remainder of this lesson, we’ll be using the most popular distance metric: the Euclidean distance.

#Hyperparameters for KNN

#1. The value of K
#2. The distance metric to use

# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn
 
# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.cross_validation import train_test_split
 
# otherwise we're using at lease version 0.18
else:
	from sklearn.model_selection import train_test_split
 
# load the MNIST digits dataset
mnist = datasets.load_digits()
 
# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
	mnist.target, test_size=0.5, random_state=42)
 
# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
	test_size=0.1, random_state=84)
 
# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []
 
# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainData, trainLabels)
 
	# evaluate the model and update the accuracies list
	score = model.score(valData, valLabels)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)
 
# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))


#Using cross validation technique or 3 split schema(train/validation/test) technique we can identify optimal values for these hyper parameters.

#Q. When using the k-NN algorithm, no actual “learning” is performed.
#   True

#Q. The k-NN algorithm can learn from the mistakes it makes on training data.
#   False

#Q. The value of k in k-NN controls:
#   The number of k closest data points allowed to cast a vote.

#SVM Quiz:
# Q. A set of data points is considered linear separable if:
# A. If we can draw a single curved line that cleanly separates all data points from class #1 and class #2
# B. If we can draw a single straight line that cleanly separates all data points from class #1 from class #2
# C. If we can draw multiple lines that cleanly separates all data points from class #1 and class #2
#Answer: B

