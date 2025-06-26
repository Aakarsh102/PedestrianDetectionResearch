'''
hog_vector_extractor.py
Author: Donny Weintz
Date: 9/26/24

This script will take an input image and perform a window
operation that extracts HOG feature vectors for various regions
of the image. The vectors will be returned in a list and can be
passed into a SVM classifier to be classified as containing a
pedestrian or not containing a pedestrian.
'''

# create a function to exract hog features for various image regions

# args: image, dimensions of window, hog tile size, hog norm size
# return: a list of feature vectors for the regions of the image

# obtain input image

# preprocess image (grayscale, change aspect ratio?)

# convert image into array

# perform window operation

    # kernel = size of window

    # call hog function to get feature vector for the window

    # add feature vector to the list

