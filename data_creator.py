import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Inspired by JMini
Author: Aditya Mallepalli
Data: 9/29/24

This code creates data for two different classes. Will be used for SVM.
'''

def linear_data():
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
    return data
def nonlinear_data():
    # Parameters for class 0
    mean_class0 = [2, 2]  # Mean for class 0
    cov_class0 = [[1.5, 0.5], [0.5, 1.5]]  # covariance non clear seperation

    # Parameters for class 1
    mean_class1 = [6, 6]  # Mean for class 1, far apart from class 0
    cov_class1 = [[1.5, 0.5], [0.5, 1.5]]  # Same covariance as class 0

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
    return data

'''
data = nonlinear_data()
# Plot the data to visualize complete linear separation
plt.scatter(data[data['Label'] == 0]['Feature1'], data[data['Label'] == 0]['Feature2'], color='red', label='Class 0')
plt.scatter(data[data['Label'] == 1]['Feature1'], data[data['Label'] == 1]['Feature2'], color='blue', label='Class 1')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Completely Linearly Separable Classes')
plt.show()

# Display the first few rows of the dataset
print(data.head())
'''
