# This program takes digits (0-9) in the mnist.mat and attempts to correctly
# guess what they are using nearest neighbor classification.

import scipy.io as sio
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

# Initialize our train and test sizes
train_size = 500
test_size = 500

# This creates the confusion matrix from the list given
def confusion_matrix(e_conf):
    plt.figure(figsize=(10, 10))
    plt.matshow(e_conf)
    plt.colorbar()
    plt.title('Confusion Matrix As Distribution')

# This creates the histogram plot from the histogram list
def hist_plot(hist):
    plt.figure()
    plt.hist(hist, bins=np.arange(11) - 0.5, density=True, rwidth=.5)
    plt.title('Likelihood of Confusion')
    plt.xlabel('Digit')
    plt.ylabel('Probability of Misclassification')
    
# Load the image data
mat = sio.loadmat("mnist.mat")

# Assign test and train images to their respectively named variables
# / 255.0 normalizes the pixels in each image
train_image = mat['trainX'] / 255.0
train_image = train_image[:train_size]
train_label = mat['trainY']
train_label = train_label[:train_size]

test_image = mat['testX'] / 255.0
test_image = test_image[:test_size]
test_label = mat['testY']
test_label = test_label[:test_size]

# Variables to hold the number of correct guesses
euclid_guess = 0
cosine_guess = 0

# Lists to hold data for confusion matrix and histogram
euclid_confusion = np.zeros((10, 10))
histogram = []

# j gives us a way to keep track of which index we should be looking at in the test_label
# and where it corresponds in our other lists
j = 0

# Loop through and calculate the euclidean and cosine distances for each image in the test set
for test_im in test_image:
    # A list to hold the euclidean distance of each train image to the test image
    euclid_distances = [np.sqrt(np.sum((image - test_im)**2)) for image in train_image]
    
    # Fine the smallest distance from the list
    min_index = np.argmin(euclid_distances)
    
    # If the train image label equals the test image label then it was guessed
    # correctly. We then increase our euclid_guess by one to count one success
    if train_label[0, min_index] == test_label[0, j]:
        euclid_guess += 1
        
    # Otherwise increase the cell of our guessed image and correct image by 1
    # to increase the pixel brightness for our confusion matrix.
    # We also add the incorrectly guessed number to our histogram list
    else:
        euclid_confusion[test_label[0, j]][train_label[0, min_index]] += 1
        histogram.append(train_label[0, j])
    
    # This does the same basically as those notes listed above
    cosine_distances = [distance.cosine(image, test_im) for image in train_image]
    min_index = np.argmin(cosine_distances)
    
    if train_label[0, min_index] == test_label[0, j]:
        cosine_guess += 1
    j += 1

# Calculate and print the accuracy of each distance algorithm
print('Euclid Accuracy: {:.2g}%'.format(euclid_guess / test_size * 100))
print('Cosine Accuracy: {:.2g}%'.format(cosine_guess / test_size * 100))

# Loop through each row and alter the numbers to be a distribution
# of the numbers across that row
euclid_confusion = [row / sum(row) for row in euclid_confusion if sum(row) != 0]

# Call these methods to print the confusion matrix and the histogram plot
confusion_matrix(euclid_confusion)
hist_plot(histogram)
