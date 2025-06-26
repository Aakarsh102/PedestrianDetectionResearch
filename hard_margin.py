"""
hard_margin.py
Author: Aditya Mallepalli
Date: 10/10/24

This script creates a hard margin svm using the sklearn model
"""
from sklearn import svm
from sklearn.model_selection import train_test_split
from data_creator import linear_data
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# Calling the data creation function from my other class
data = linear_data()

#Split the data into features (X) and labels (y)
X = data[['Feature1', 'Feature2']]
y = data['Label']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)

# Creating a hard margin classifer
hard_margin = svm.SVC(kernel='linear')

# Training the model using the training set created earlier
hard_margin.fit(X_train, y_train)

# Strong the prediction using one of the testing sets created earlier
y_pred = hard_margin.predict(X_test)

# Testing the accuracy, precision and recall of the prediction
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

'''
Decision boundary plotting inspired by Donny
'''
# getting w and b values for decision boundary
w = hard_margin.coef_[0]
b = hard_margin.intercept_[0]

x_min, x_max = X['Feature1'].min() - 1, X['Feature1'].max() + 1
xx = np.linspace(x_min, x_max, 100)

# descision boundary
boundary = (-w[0] * xx - b) / w[1] 

# support vectors
pos = (-w[0] * xx - b + 1) / w[1]
neg = (-w[0] * xx - b - 1) / w[1]

# plotting the boundaries
plt.plot(xx, boundary, color = 'black', label = 'Decision Boundary')
plt.plot(xx, pos, 'b--', label = 'Positive Support')
plt.plot(xx, neg, 'r--', label = 'Negative Support')

#plotting the data points
plt.scatter(data[data['Label'] == 0]['Feature1'], data[data['Label'] == 0]['Feature2'], color='red', label='Class 0')
plt.scatter(data[data['Label'] == 1]['Feature1'], data[data['Label'] == 1]['Feature2'], color='blue', label='Class 1')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Completely Linearly Separable Classes')

plt.show()
