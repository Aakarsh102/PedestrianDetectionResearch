'''
svm_model.py
Author: Donny Weintz
Date: 9/26/24

This script is a template that aims to use SVMs to classify
HOG feature vectors as either containing pedestrians or not
containing pedestrians.
'''

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from svm_hard_margin import create_ls_data

'''
train_svm_model

This function takes in a list of hog feature vectors and a list
of correspondings labels for each feature vector in the list. The
function then splits data into training and testing sets. It trains
the svm model based on the training data and training labels. The function
returns the svm_model and the accuracy.

args:
    hog_features_list (list of vectors of same dimension)
        A list of hog feature vectors
    labels_list (list of ints)
        A list of integers. 1 = pedestrian in image, -1 = no pedestrian
return:
    svm_model (sklearn.svm._classes.SVC)
        The svm model
    acc
        The computed accuracy of the model
'''
def train_svm_model(hog_features_list, labels_list):
    # split the hog vector data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(hog_features_list, labels_list)

    # initialize svm classifier
    svm_model = svm.SVC(kernel = 'linear')

    # train the SVM classifier
    svm_model.fit(X_train, y_train)

    # test the classifier
    y_pred = svm_model.predict(X_test)
    
    # compute accuracy score for classifier
    acc = accuracy_score(y_test, y_pred)
    print(f'SVM accuracy: {acc}')
    print(svm_model)
    print(type(svm_model))

    return svm_model, acc

'''
test_vectors = [[1, 2, 3], [4, 5, 6], [-1, 2, -1], [5, -4, 5]]
test_labels = [1, 1, -1, -1]
train_svm_model(test_vectors, test_labels)
'''