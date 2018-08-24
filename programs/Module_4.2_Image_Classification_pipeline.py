#4.2: The image classification pipeline
 
#Five steps of the image classification pipeline:
#Step 1: Structuring your initial dataset.
#Step 2: Splitting the dataset into two (optionally three) parts.
#Step 3: Extracting features.
#Step 4: Training your classification model.
#Step 5: Evaluating your classifier.
 
#Step 1: Gathering your dataset
#To develop a supervised Machine Learning algorithm, we need images as well as the labels associated with each image.
#For example, our target labels could be: {cat, cow, dog, horse, wolf}
 
#Furthermore, the number of images for each category should be fairly uniform (i.e. the same).
#If we have twice the number of cat images than dog images, and five times the number of horse images than cat images,
#then our classifier will become naturally biased to “overfitting” into these heavily-represented categories.
#In order to fix this problem, we normally sample our dataset so that each category is represented equally.
 
#Step 2: Splitting the dataset into two (optionally three) parts
#We usually keep (20% or 25% or 33%) of data aside as test data, to evaluate our models before deploying them in production.
#We have something called hyper parameters (like the learning rate or number of trees to be grown for a random forest). The
#Machine learning algorithm will not optimize these parameters, and it is the responsibility of the programmer to find these optimal
#hyper parameters to be used to develop the predictive model. We have two methods to tune hyper parameters.
#(1). Cross validation methods
#(2). Validation data set
 
#In cross validation we divide our training data set into k parts (k can be 2 or more), develop model on k-1 parts and test the model on the left over part
#We repeat this process for all the k parts, and measure the average prediction error. The process is repeated for various hyper-parameter combinations
#and ultimately we will select the hyper-parameters which have given the least average prediction error.
#Usually Cross validation method is computationally expensive.
 
#In Validation data set approach, we divide the training data into 80%:20% (training:validation), develop the model on 80% of the data (using various hyper parameter combinations)
#and use 20% to test the performance of the model. Finally we choose the set of hyper-parameters which have given the least prediction error for the validation data.
 
#For other steps refer to pyimagesearchgurus.com