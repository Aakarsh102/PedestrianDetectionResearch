"""
For testing HOG, we'll be implementing 2 simple SVC classifier. One being trained on the images
themselves and the the other being trained on the HOG features.

We'll record the training time and testing accuracy for both to evaluate the performance of HOG.
"""

# Implementation starts here. (in progess...)


"""
for downloading the dataset 

***get a kaggle.json file from your kaggle account and place it in the ~/.kaggle directory.***


Then run the following code to download the dataset
kaggle datasets download -d aliasgartaksali/human-and-non-human

"""

import numpy as np
import cv2
import pandas as pd
import os
from HOG import create_histograms
from block_normalizer import block_normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

""" This is a modified version of dweintz's HOG implementation"""


def do_HOG(data, cell_size ):
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    feature_descriptors = []

    # Get image dimensions
    for img_arr in data:
        height, width = img_arr.shape
        # Initialize gradient arrays
        gradient_x = np.zeros_like(img_arr, dtype=float)
        gradient_y = np.zeros_like(img_arr, dtype=float)
        dir = np.zeros_like(img_arr, dtype=float)

                # Loop through every pixel in the image
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = img_arr[i - 1:i + 2, j - 1:j + 2]
                gradient_x[i, j] = np.sum(region * sobel_x)
                gradient_y[i, j] = np.sum(region * sobel_y)
                if gradient_x[i, j] != 0:
                    dir[i, j] = math.degrees(math.atan(gradient_y[i, j] / gradient_x[i, j]))
                else:
                    dir[i, j] = 90

                # Calculate gradient magnitude and direction
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        gradient_magnitude = gradient_magnitude *2
        gradient_magnitude = gradient_magnitude.astype(np.uint8)

        

                # Initialize the HOG feature vector
        cell_size = 8  # 8x8 cells
        bins = 9  # 9 bins for 0-180 degrees
    
        HOG_feature_vector = []

                # Process 8x8 cells
        for i in range(0, height, cell_size):
            for j in range(0, width, cell_size):
                        # Initialize histogram for the current cell
                hist = [
                    [0, 0],
                    [20, 0],
                    [40, 0],
                    [60, 0],
                    [80, 0],
                    [100, 0],
                    [120, 0],
                    [140, 0],
                    [160, 0]
                ]

                        # Loop through each pixel in the 8x8 cell
                for x in range(cell_size):
                    for y in range(cell_size):
                        angle = dir[i + x, j + y]
                        magnitude = gradient_magnitude[i + x, j + y]

                                # Determine which bins the angle falls between
                        lower_bin = int(angle / 20)
                        upper_bin = (lower_bin + 1) % bins  # Wrap around to bin 0 if needed

                                # Calculate the contribution to the bins
                        bin_fraction = angle / 20 - lower_bin
                        hist[lower_bin][1] += magnitude * (1 - bin_fraction)
                        hist[upper_bin][1] += magnitude * bin_fraction

                        # Append the histogram for this cell to the feature vector
                HOG_feature_vector.append(hist)  # Append the hist for each cell only once

        l = [np.array([k[1] for k in i]) for i in HOG_feature_vector]
        feature_descriptor = block_normalize(l, 4)
        feature_descriptors.append(feature_descriptor.flatten())    
            # Convert HOG feature vector to a NumPy array
            # HOG_feature_vector = np.array(HOG_feature_vector)
    return feature_descriptors

import zipfile
zip_ref = zipfile.ZipFile('human-and-non-human.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()

x_val = []
y_val = []

# I'm creating the train set here
count = 0
for i in os.listdir("human-and-non-human/training_set/training_set/humans"):
    if (count == 1000):
        break
    count += 1
    arr = cv2.imread("human-and-non-human/training_set/training_set/humans/" + i)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, [128, 128])
    # arr = cv2.resize(arr, [32, 64])
    # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB
    x_val.append(np.array(arr))
    y_val.append(1)
count = 0
for i in os.listdir("human-and-non-human/training_set/training_set/non-humans"):
    if (count == 1000):
        break
    count += 1
    arr = cv2.imread("human-and-non-human/training_set/training_set/non-humans/" + i)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, [128, 128])
    # arr = cv2.resize(arr, [32, 64])
    # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB
    x_val.append(np.array(arr))
    y_val.append(0)

# I've done this to make sure that all the images are of the same size
# it's a sanity check 
for i in x_val:
    if (i.shape != x_val[0].shape):
        print(i.shape)

x_val = np.array(x_val)
y_val = np.array(y_val)


x_test = []
y_test = []

# I'm creating the test set here 
count = 0
for i in os.listdir("human-and-non-human/test_set/test_set/humans"):
    if (count == 200):
        break
    count += 1
    arr = cv2.imread("human-and-non-human/test_set/test_set/humans/" + i)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, [128, 128])
    # arr = cv2.resize(arr, [32, 64])
    # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB
    x_test.append(np.array(arr))
    y_test.append(1)
count = 0
for i in os.listdir("human-and-non-human/test_set/test_set/non-humans"):
    if (count == 200):
        break
    count += 1
    arr = cv2.imread("human-and-non-human/test_set/test_set/non-humans/" + i)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, [128, 128])
    # arr = cv2.resize(arr, [32, 64])
    # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB
    x_test.append(np.array(arr))
    y_test.append(0)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = []
np.random.shuffle(x_val)
x_train = do_HOG(x_val, 8)

svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(np.array(x_train), y_val)

x_test_new = []
x_test_new = do_HOG(x_test, 8)

reshaped_arr = np.array(x_test_new).reshape(1, -1)

y_pred = svm_classifier.predict(reshaped_arr)

print(accuracy_score(y_test, y_pred))