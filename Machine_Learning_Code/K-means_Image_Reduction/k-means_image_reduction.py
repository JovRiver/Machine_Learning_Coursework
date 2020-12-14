# This program uses K-means to reduce the number of individual colors in the given images.

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# Load all of the images we want
im1 = imread('https://s3-us-west-2.amazonaws.com/uw-s3-cdn/wp-content/uploads/sites/6/2017/11/04133712/waterfall.jpg')
im2 = imread('http://www.youandthemat.com/wp-content/uploads/nature-2-26-17.jpg')
im3 = imread('https://buddhisteconomics.net/wp-content/uploads/2017/10/hdwp693968124.jpg')
im4 = imread('https://upload.wikimedia.org/wikipedia/commons/d/dc/Skyscrapers_of_Shinjuku_2009_January_%28revised%29.jpg')
im5 = imread('https://www.worldatlas.com/r/w728-h425-c728x425/upload/db/04/18/toronto-ontario.jpg')
im6 = imread('https://www.valdosta.edu/academics/online-programs/images/slideshow-new-vsu-arch.jpg')
im7 = imread('https://www.valdosta.edu/administration/planning/images/planningimage.jpg')
im8 = imread('http://www.carterusa.com/project-images/develop/program-Project-Management/higher-education/Valdosta-State-Student-Housing/gal-1.JPG')

# Set k value for kmeans
k = 2
# Set im equal to the image we want to alter
im = im1

# Keep track of the original images shape
im_shape = np.shape(im)

# Perform resizing on the image to make calculations faster
# x is our width and y is our height
x = 256
y = int(x / (im_shape[1] / im_shape[0]))
# Resize and reshape the image to an x*y by 3 vector
im_r = np.squeeze(np.reshape(resize(im, (y, x)), (-1, 1, 3)))

# Find the length of our reshaped image vector
N = len(im_r)

# Assign random values for our mu's
mus = np.array([im_r[np.random.randint(N)] for i in range(k)])

# Re-calculate our mu's to find the cluster center
for ix in range(10):
    r = np.zeros((N, k))
    # Loop through every vector in our im_r and find the closest pixels
    for i in range(N):
        dists = [np.sqrt(np.sum(mus[j] - im_r[i])**2) for j in range(k)]
        # Assign a 1 for the column value that is the closest to our mu
        r[i, np.argmin(dists)] = 1
    # Adjust our mu's
    mus = np.array([np.dot(r[:, i], im_r) / np.sum(r[:, i]) if np.sum(r[:, i]) != 0 else np.zeros(3) for i in range(k)])
    
# Change each pixel value to be the same as the closest mu's value
im_r = [mus[np.argmin([np.sqrt(np.sum(mus[j,:] - im_r[i])**2) for j in range(k)])] for i in range(N)]

# Reshape and plot our new image
im_r = np.reshape(im_r, (y, x, 3))
plt.imshow(im_r)
