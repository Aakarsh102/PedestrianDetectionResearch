'''
Author: Aditya Mallepalli
Data: 10/14/24

This is the actual pedestrian detector model created from HOG
feature vectors being fed into a SVM.
'''

# from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import extract_pedestrians_and_subimages
from joblib import dump
import cv2

#call the function to extract pedestrians and the subimages with no pedestrians
cropped_images = extract_pedestrians_and_subimages('scrubbed_train_bbox.txt', '/Users/amallepalli/Desktop/ad_01', 10000, 0.5)
feature_vectors = []
labels = []
hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
#loop through the cropped images so that we can call HOG on them and get the feature vectors
for i in range(len(cropped_images)):
    #print('iteration:', i)
    #feature vectors of each image being appended to list
    # feature_vectors.append(hog(cropped_images[i][0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False))
    current_vectors = hog.compute(cropped_images[i][0])
    feature_vectors.append(current_vectors)
    #label of each image being appeneded to list (-1 or 1)
    labels.append(cropped_images[i][1])
print('Finished HOG')
#convert the feature vectors and labels to numpy arrays that will be used for training and testing
X = np.array(feature_vectors)
y = np.array(labels)

#create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)
print('Finished splitting Data')

#create an rbf svm classifier
svm_classifier = svm.SVC(kernel='rbf')

#train the classifier with our data
svm_classifier.fit(X_train, y_train)
print('Finished Training')
#store our prediction
y_pred = svm_classifier.predict(X_test)
print('Finished Predicting')

# Testing the accuracy, precision and recall of the prediction
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

#save the model
dump(svm_classifier, 'pedestrian_detector.joblib')