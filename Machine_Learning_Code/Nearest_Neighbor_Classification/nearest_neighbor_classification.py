# This program uses nearest neighbor classification to attempt to classify images of foliage taken
# from satellite imagery into diseased or non-diseased foliage.

import csv
import numpy as np
from collections import Counter

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

file = open('training.csv')
csv_reader = csv.reader(file)
train_set = [data for data in csv_reader][1:]

file = open('testing.csv')
csv_reader = csv.reader(file)
test_set = [data for data in csv_reader][1:]

#########################################################################################
# Given test/train split #

# Training and test set features
X_train = np.array([features[1:] for features in train_set]).astype(np.float64)
X_test = np.array([features[1:] for features in test_set]).astype(np.float64)

# Training and test set labels
y_train = np.array([label[0] for label in train_set])
y_test = np.array([label[0] for label in test_set])

# Set up the K-Nearest-Neighbor object and fit the training set
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Find our predictions
y_pred = knn.predict(X_test)

# Find the number of positive hits for both diseased trees and non-diseased foliage
tree_hit = sum([1 for i in range(len(y_test)) if y_pred[i] == 'w' and y_test[i] == 'w'])
fol_hit = sum([1 for i in range(len(y_test)) if y_pred[i] == 'n' and y_test[i] == 'n'])

# Count the occurrence of each type in the test set
counts = dict(Counter(y_test))
# Create a confusion matrix using the hit counts and the total counts from counts
confusion = [[tree_hit, counts.get('w') - tree_hit], [counts.get('n') - fol_hit, fol_hit]]

# Print the accuracy results and the confusion matrix
print('Accuracy: {:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred)))
print('Confusion Matrix:\n{:}\n'.format(confusion))

#########################################################################################
# Personal test/train split #

# Note: there is a disproportionate share of non-wilted foliage and the authors
#       split the data set with more wilted tress in the test set than the training
#       set so I altered the data to allow us to choose the percentage of wilted and
#       non-wilted foliage in the training set

# Choose percentage of wilted and non-wilted foliage in our training set
data_set = np.array([features[1:] for features in train_set] + [features[1:] for features in test_set]).astype(np.float64)
labels = np.array([label[0] for label in train_set] + [label[0] for label in test_set])

X_train, X_test, y_train, y_test = train_test_split(data_set, labels, test_size=0.1)

# I referenced this link to learn how to use the KNearestNeighbor function from sklearn
# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

# Set up the K-Nearest-Neighbor object and fit the training set
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Find our predictions
y_pred = knn.predict(X_test)

# Find the number of positive hits for both diseased trees and non-diseased foliage
tree_hit = sum([1 for i in range(len(y_test)) if y_pred[i] == 'w' and y_test[i] == 'w'])
fol_hit = sum([1 for i in range(len(y_test)) if y_pred[i] == 'n' and y_test[i] == 'n'])

# Count the occurrence of each type in the test set
counts = dict(Counter(y_test))
# Create a confusion matrix using the hit counts and the total counts from counts
confusion = [[tree_hit, counts.get('w') - tree_hit], [counts.get('n') - fol_hit, fol_hit]]

# Print the accuracy results and the confusion matrix
print('Accuracy: {:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred)))
print('Confusion Matrix:\n{:}\n'.format(confusion))


