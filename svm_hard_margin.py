'''
svm_hard_margin.py
Author: Donny Weintz
Date: 10/7/24

This script implements hard margin svm manually by performing the quadratic
programming problem associated with hard margin svm.
'''

import matplotlib.pyplot as plt                     # graphing/display
import numpy as np                                  # matrices
from cvxopt import matrix, solvers                  # optimizations for svm

'''
create_ls_data

This function takes inspiration from Jmini's generate_classes function
in svm_jw.py. It takes in various attributes for the data and randomly
generates 2 linearly separable classes.

args:
    num_points (int)
        The number of points for each class.
    c1_mean (tuple of ints)
        The (x, y) coordinates of class 1 mean.
    c2_mean (tuple of ints)
        The (x, y) coordinates of class 2 mean.
return:
    class1 (array)
        Array containing the coordinates of points in class 1
    class2 (array)
        Array containing the coordinates of points in class 1
    class1_labels (array)
        Array containing labels for class 1 (+1)
    class2_labels (array)
        Array containing labels for class 2 (-1)
'''
def create_ls_data(num_points, c1_mean, c2_mean, cov):
    # generate random points for the classes
    class1 = np.random.multivariate_normal(c1_mean, cov, num_points)
    class2 = np.random.multivariate_normal(c2_mean, cov, num_points)

    # generate labels (+1 for class 1, -1 for class 2)
    class1_labels = np.ones(num_points)
    class2_labels = np.ones(num_points) * -1

    return class1, class2, class1_labels, class2_labels

'''
hard_margin_svm(X, Y)

This function uses the CVXOPT library to solve the optimization
for hard margin svm.

args:
    X (array)
        Array containing the feature vectors
    Y (array)
        A vector containing the label for each feature vector
return:
    w (array)
        Array containing the weights for the optimal hyperplane
    b (float)
        Value of b for the optimal hyperplane
'''
def hard_margin_svm(X, Y):
    # number of data points
    num_samples = len(Y)
 
    # create P matrix - quadratic part (y_i * y_j * (x_i dot x_j))
    P = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            P[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

    # convert P to a cvxopt matrix
    P = matrix(P, (P.shape[0], P.shape[1]), 'd')

    # q vector (negative of ones)
    q = matrix(-np.ones(num_samples))

    # G matrix, h vector - inequality constraint (alpha_i >= 0)
    G = matrix(np.diag(-np.ones(num_samples)))
    h = matrix(np.zeros(num_samples))

    # A matrix, b - equality constraint (sum(alpha_i * y_i) = 0)
    A = matrix(Y, (1, num_samples), 'd')
    b = matrix(0.0)

    # solve the quadratic programming problem
    solution = solvers.qp(P, q, G, h, A, b)

    # the optimized lagrange values
    alphas = np.array(solution['x'])

    # solve for w
    w = np.sum(alphas * Y[:, None] * X, axis = 0)
    
    # solve for b
    support_vector_indices = np.where(alphas > 1e-5)[0]
    b = Y[support_vector_indices[0]] - np.dot(w, X[support_vector_indices[0]])
   
    return w, b

'''
plot_hard_margin_svm(2, b, X)

This function plots the hyperplane and support vectors for
hard margin SVM.

args:
    w (array)
        Array containing the weights for the optimal hyperplane
    b (float)
        Value of b for the optimal hyperplane
    X (array)
        Array containing the feature vectors
return:
    None
'''
def plot_hard_margin_svm(w, b, X):
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
    


# TEST FUNCTIONS WITH 2-D DATASET

# create a data set of linearly separable data with class labels
num_points = 100           # number of points for each class
c1_mean = (1, 1)           # (x, y) of class 1 mean
c2_mean = (7, 7)           # (x, y) of class 2 mean
cov = [[1, 0], [0, 1]]     # covariance
np.random.seed(5)

# call function to create linearly separable data
class1, class2, class1_labels, class2_labels = create_ls_data(num_points, c1_mean, c2_mean, cov)

# clean up data before sending into hard_margin_svm
X = np.concatenate((class1, class2), axis = 0)
Y = np.concatenate((class1_labels, class2_labels), axis = 0)

# obtain the w vector and b
w, b = hard_margin_svm(X, Y)

print(w)
print(b)

# visualize the results
plt.figure(figsize = (10, 7))

plot_hard_margin_svm(w, b, X)

plt.title('Hard Margin SVM')
plt.scatter(class1[:, 0], class1[:, 1], color = 'blue', label = 'Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color = 'red', label = 'Class 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

