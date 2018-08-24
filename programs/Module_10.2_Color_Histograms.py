#10.3 Color histograms
#Color histogram represents the histogram of pixels intensities in the image. 
#Unlike a the mean and standard deviation which attempt to summarize the pixel intensity distribution, 
#a color histogram explicitly represents it! In fact, a color histogram is the color distribution!

#We’ll still be operating under the same assumption that images with similar color distributions contain equally 
#similar visual contents — this assumption may or may not hold in your particular application. 
#For many computer vision systems (especially for those operating under controlled lighting conditions) 
#color histograms become an extremely valuable and powerful image descriptor.

#A color histogram counts the number of times a given pixel intensity (or range of pixel intensities) occurs in an image. 
#Using a color histogram we can express the actual distribution or “amount” of each color in an image. 
#The counts for each color/color range are then used as our feature vector.

