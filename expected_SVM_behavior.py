import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# loss function being used here is l = 2 * lambda * ||w||^2 + \sum max(0, 1 - y_i(wx + b)))
# I'm using gradient descent algorithm


# Data generation code (was generated using ChatGPT),
# this code block is being used by the entire team.

# Parameters for class 0
mean_class0 = [2, 2]  # Mean for class 0
cov_class0 = [[0.5, 0], [0, 0.5]]  # Reduced covariance for clear separation
 
# Parameters for class 1
mean_class1 = [6, 6]  # Mean for class 1, far apart from class 0
cov_class1 = [[0.5, 0], [0, 0.5]]  # Same covariance as class 0
 
# Generate data
np.random.seed(42)  # For reproducibility
class0_data = np.random.multivariate_normal(mean_class0, cov_class0, 100)
class1_data = np.random.multivariate_normal(mean_class1, cov_class1, 100)
 
# Create labels
labels_class0 = np.zeros(100)
labels_class1 = np.ones(100)
 
# Combine data into a pandas DataFrame
data_class0 = pd.DataFrame(class0_data, columns=['Feature1', 'Feature2'])
data_class0['Label'] = labels_class0
 
data_class1 = pd.DataFrame(class1_data, columns=['Feature1', 'Feature2'])
data_class1['Label'] = labels_class1
 
# Combine both classes into one DataFrame
data = pd.concat([data_class0, data_class1], ignore_index=True)

# this is to get the numpy arrays of the x and y values for
x_train = data.iloc[:, :-1].values
y_train = data.iloc[:, -1].values

# The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.
SupportVectorClassifier = SVC(kernel = 'linear', C = 0.1)   
print("Training the model...")
SupportVectorClassifier.fit(x_train, y_train)
print("Model trained successfully!")
w = SupportVectorClassifier.coef_
b = SupportVectorClassifier.intercept_
# This function has been taken for sklearn_soft_margin.py file in the same repo. 
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
    plt.scatter(data[data['Label'] == 0]['Feature1'], data[data['Label'] == 0]['Feature2'], color='red', label='Class 0')
    plt.scatter(data[data['Label'] == 1]['Feature1'], data[data['Label'] == 1]['Feature2'], color='blue', label='Class 1')
    plt.show()

print("Visualizing the decision boundary...")
# visualize the decision boundary 
plot_soft_margin_svm(w[0], b[0], x_train)