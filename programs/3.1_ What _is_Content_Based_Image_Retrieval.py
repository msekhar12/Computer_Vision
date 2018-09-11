#3.1: What is Content-Based Image Retrieval?
#"Image search engines" encompass methods to make a dataset of images visually searchable using only the contents of the image.
#Academics call this Content-Based Image Retrieval (CBIR).
#The terms “image search engine” and “CBIR” are used interchangeably.
 
#At the core, image search engines rely on extracting features from images and then comparing the images for similarity based
#on the extracted feature vectors and distance metric. CBIR also includes methods to:
# 1. Efficiently store features extracted of images.
#2. Scale the time it takes to perform a search logarithmically as the size of the image dataset increases linearly.
#3. Combine techniques from computer vision, information retrieval, and databases to build real-world images search engines that can be deployed online.
 
#Search engines like Google are text based search engines. You enter text to be searched and Google returns the best results.
#Image search engine takes an image as an input and returns the images which are closely related to the image being searched.
 
#In general, there tend to be three types of image search engines:
#1. search by meta-data
#2. search by example, and
#3. a hybrid approach of the two.
 
#1. Search by meta-data systems rarely examine the contents of the image itself.
#Instead, they rely on textual clues such as (1) manual annotations and tagging performed by humans along with (2) automated contextual hints, such
#as the text that appears near the image on a webpage.
#User enters a search query in the form of a string, and the images are returned which have the matching meta data as the response.
#Example: Flickr. After uploading an image to Flickr you are presented with a text field to enter tags describing the contents of images
#you have uploaded. Flickr then takes these keywords, indexes them, and utilizes them to find and recommend other relevant images.
 
#2. Search by example systems, on the other hand, rely solely on the contents of the image. NO keywords are assumed to be provided.
#Image search engines that quantify the contents of an image are called Content-Based Image Retrieval (CBIR) systems.
#Example: TinEye
#Important to reinforce the point that Search by Example systems rely strictly on the contents of the image and nothing else.
 
#In hybrid approach you could build an image search engine that could take both contextual hints along with a Search by Example strategy.
 
#Important terms:
#Features (or feature vectors) are just a list of numbers used to abstractly represent and quantify images.
#Image descriptor converts an image into a feature vector globally. That is the whole image is quantified into a feature vector
#To build image search engine (CBIR), we extract the features of images and store them so that an input image is
#quantified as a feature vector and these features are matched (using some similarity metric such as eucledian distance or cosine similarity etc.)
#against the stored features and the best matched images are returned.
 
#See https://github.com/msekhar12/Computer_Vision/blob/master/programs/Module_10.1_Image_Descriptors.py for more information about image descriptors
#In some cases, indexing our dataset is as simple as storing our feature vectors in a .csv file.
#But for large-scale image search engines, we use specialized data structures and algorithms to make searches in an
#N-dimensional space run in sub-linear time.
 
#4 Steps for developing a CBIR system:
#1. Define your image descriptor: Are you interested in the color of the image? The shape of an object in the image? Or do you want to characterize texture?
#2. Feature extraction and indexing your dataset: Apply this image descriptor to each image in your dataset, extract features from these images,
#   and write the features to storage (ex. CSV file, RDBMS, Redis, etc.) so they can later be compared for similarity.
#   Furthermore, you need to consider if any specialized data structures are going to be used to facilitate faster searching.
#3. Defining your similarity metric: Popular choices include the Euclidean distance, Cosine distance,
#   and chi-square distance, but the actual choice is highly dependent on (1) your dataset and (2) the types of features you extracted.
#4. Searching: The final step is to perform an actual search. A user will submit a query image to your system (from an upload form or via a mobile app,
#   for instance) and your job will be to (1) extract features from this query image and then (2) apply your similarity function to compare the query features
#   to the features already indexed. From there, you simply return the most relevant results according to your similarity function.
 
 
#Steps 1 and 2:
# Dataset of Images ===> Extract features from each image  ===> Store features
 
#Steps 3 and 4:
# User submits query image ===> Extract features of query image ===> Compare query features to image features  <=== Database of features
#                                                                                 |
#                      display results to user <=== Sort results by relevance <===|
 
 
#Measuring the performance of CBIR:
#We can use the following metrics to measure the performance of CBIR:
 
#Precision = Number of relevant images retrived / Total number of images retrieved from database
 
#Recall = Number of relevant images retrived / Total number of relevant images in the database
 
#F1-score = (2 X Precision X Recall)/(Precision + Recall)
 
 
#Obviously, the assumption when using these metrics is that we know how many relevant images there are
#to a particular query in our dataset. This works well in academic and research datasets where CBIR systems must be
#benchmarked evaluated and measured against a standard dataset with known ground-truth result sets.
 
#But in many real-world CBIR problems, it may not be possible for us to know exactly how many relevant images that are
#to a particular query. Instead, we must (1) visually investigate the result sets from particular queries,
#(2) utilize our general intuition when deciding what is working well and what is not, and
#(3) create small “test sets” for particular images that we can use to measure the accuracy of our CBIR system.
 
#Very important:
#Is Machine Learning and CBIR same or different?
#They are different. Machine learning deals with automatically learning features to classify an image.
#But in CBIR there is no training involved. We are just getting the similarity metric between the image being queried and
#the images inside the images database.
 
 
#Q. All of the following are types of image search engines EXCEPT:
#A. Search by meta-data
#B. Search by example
#C. Hybrid approach
#D. Search by canonical
#Answer: D
 
#Q. Search by example CBIR systems analyze both the content of the image and any text, tags, and/or meta-data associated with the image:
#A. False
 
#Q. The following are all required steps in building a CBIR system EXCEPT:
#A. Defining an image descriptor
#B. Feature extraction and indexing
#C. Searching
#D. Defining a similarity metric/distance function
#E. Spatial verification
#Answer: E
 
#Q. Unlike supervised machine learning, CBIR systems assume no concept of “labeled training data” and can instead be viewed as a simple yet “dumb” machine learning classifier:
#A. True