'''
sklearn_hard_margin.py
Author: Donny Weintz
Date: 10/9/24

This script implements soft margin svm on a linearly inseparable dataset.
'''

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


def train_svm_model(X, Y, reg_param):
    # initialize svm classifier
    svm_model = svm.SVC(C = reg_param, kernel = 'linear')

    # train the SVM classifier
    svm_model.fit(X, Y)
    
    return svm_model

def create_ls_data(num_points, c1_mean, c2_mean, cov):
    # generate random points for the classes
    class1 = np.random.multivariate_normal(c1_mean, cov, num_points)
    class2 = np.random.multivariate_normal(c2_mean, cov, num_points)

    # generate labels (+1 for class 1, -1 for class 2)
    class1_labels = np.ones(num_points)
    class2_labels = np.ones(num_points) * -1

    return class1, class2, class1_labels, class2_labels

def plot_soft_margin_svm(w, b, X):
     # generate values for axis
    x1 = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
    
    # use w and b to get decision boundary
    boundary = (-w[0] * x1 - b) / w[1] 
    
    # find support vectors
    pos_support_vector = (-w[0] * x1 - b + 1) / w[1]
    neg_support_vector = (-w[0] * x1 - b - 1) / w[1]
    
    # plot the decision boundary and support vectors
    plt.plot(x1, boundary, color = 'black', label = 'Decision Boundary')
    plt.plot(x1, pos_support_vector, 'b--', label = 'Positive Support')
    plt.plot(x1, neg_support_vector, 'r--', label = 'Negative Support')
    
# create a data set of linearly separable data with class labels
num_points = 100           # number of points for each class
c1_mean = (1, 1)           # (x, y) of class 1 mean
c2_mean = (3.5, 3.5)       # (x, y) of class 2 mean
cov = [[1, 0], [0, 1]]     # covariance
np.random.seed(5)          # set a seed for reproducibility

# call function to create linearly separable data
class1, class2, class1_labels, class2_labels = create_ls_data(num_points, c1_mean, c2_mean, cov)

# clean up data before sending into hard_margin_svm
X = np.concatenate((class1, class2), axis = 0)
Y = np.concatenate((class1_labels, class2_labels), axis = 0)
reg_param = 1

# train the svm hard margin model
svm_model = train_svm_model(X, Y, reg_param)

w = svm_model.coef_[0]
b = svm_model.intercept_[0]

print(w)
print(b)

# visualize the results
plt.figure(figsize = (10, 7))

plot_soft_margin_svm(w, b, X)

plt.title('Soft Margin SVM')
plt.scatter(class1[:, 0], class1[:, 1], color = 'blue', label = 'Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color = 'red', label = 'Class 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

