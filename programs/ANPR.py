#4 steps required to build any ANPR (Automatic Number Plate Recognition) system

#Step 1: Photo acquisition
#The first step in any ANPR system is to actually acquire a photo of a car that (presumably) has a license plate that we want to detect and recognize.
#Most production-level ANPR systems deployed in the real-world utilize infrared cameras so photos can be captured of vehicles regardless of time of day.
#There are a variety of endless ways to capture our original image of a vehicle. 
#But the point here is that we must (1) consider our surroundings, (2) determine which camera/setup will work best, and (3) deploy our camera in the wild.

#Step 2: Localization
#From the image of the car we need to find or localize the region(s) of the image that contain the license plate(s).
#There are a variety of methods to accomplish the localization tasks. 
#Perhaps surprisingly, we do not need any fancy machine learning algorithms to detect license plates in images 
#(although machine learning can help in certain situations).
#In general, we can leverage clever combinations of basic image processing techniques to reveal regions of an image that could contain a license plate. 
#We call these regions license plate candidates, and use further image processing to filter out the false-positives.


#Step 3: Segmentation
#In order to identify each of the characters on the license plate, we first need to segment them from the license plate background.
#In general, you’ll find that license plate images are too noisy to apply Optical Character Recognition (OCR) algorithms on directly, 
#so we’ll further need leverage our image processing skills.
#Most techniques perform some sort of adaptive thresholding or scissoring to “cut” the characters from the license plate

#Step 4: Recognition
#Finally, given each (cleanly segmented) individual character on the license plate, we apply a bit of machine learning to recognize it.

#ANPR systems are developed on a state-by-state basis. 
#Don’t be surprised if, when you go outside and snap a photo of your own car’s license plate, the license plate identification fails. 





