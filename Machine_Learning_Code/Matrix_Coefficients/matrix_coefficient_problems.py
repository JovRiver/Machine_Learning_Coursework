# This program first calculates the dot product of two matrices to determine if matrix b# is a solution
# to the linear equations given by m#

# The second part of this program calculates the polynomial coefficients of an equation of which 5 points 
# are known.

# The last part of this program will attempt to approximate the solution of an over determined system.

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# PART 1

m1 = np.array([[5, -3, 2], 
               [2, 4, -1], 
               [1, -11, 4]])

b1 = np.array([[3], 
               [7], 
               [3]])

# Question 2 Values
m2 = np.array([[1, 0, 4], 
               [4, -2, 1], 
               [2, -2, -7]])

b2 = np.array([[13], 
               [7], 
               [-19]])

# Question 3 Values
m3 = np.array([[-2, 1], 
               [1, 1]])

b3 = np.array([[-3], 
               [3]])

# Question 4 Values
m4 = np.array([[10, -7, 0], 
               [-3, 2, -6], 
               [5, 1, 5]])

b4 = np.array([[7], 
               [4], 
               [-19]])

# Question 5 Values
m5 = np.array([[1, 4, -1, 1], 
               [2, 7, 1, -2], 
               [1, 4, -1, 2],
               [3, -10, -2, 5]])

b5 = np.array([[2], 
               [16],
               [-15],
               [-15]])

# Varaibles to hold above matrices
matrices = []
b = []

matrices.append(m1)
matrices.append(m2)
matrices.append(m3)
matrices.append(m4)
matrices.append(m5)

b.append(b1)
b.append(b2)
b.append(b3)
b.append(b4)
b.append(b5)

print("Results for Part 1\n")

# loop through and perform the appropriate matrix multiplication
# with a try except to catch matrices with no solution
for i in range(5):
    try:
        matrix = np.matmul(np.linalg.inv(matrices[i]), b[i])
        print(f'Problem {i + 1} solution\n {matrix}\n')
    except:
        print(f'Problem {i + 1} has no solution\n')

##############################################################################
# PART 2

# Polynomial: f(x) = a0 + a1 * x^1 + a2 * x^2 + a3 * x^3 + a4 * x^4

# Matrix Values
matrix = [[1, -0.5, 0.25, -0.125, 0.0625], 
          [1, -0.2, 0.04, -0.008, 0.0016], 
          [1, 0.5, 0.25, 0.125, 0.0625], 
          [1, 0.75, 0.5625, 0.421875, 0.31640625],
          [1, 1, 1, 1, 1]]

# Column Bias Vector
b = [[7.625], 
     [9.3632], 
     [9.625], 
     [8.7578], 
     [8]]

print("Results for part 2\n")

# Calculate the solution to x = A^-1 * b
solution = np.matmul(np.linalg.inv(matrix), b)
print(f'Solution:\n {solution}')

print("\nGraphs for the second part of part 2 and for part 3 are given on the graph page\n")

# Lambda function to calculate the point on the line at x using the
# polynomial from above with the calculated values for w
f = lambda x, w: w[0] + w[1] * x + w[2] * (x**2) + w[3] * (x**3) + w[4] * (x**4)

# Setup the domain of the graph and calculate the values for x
domain = np.linspace(-2, 2, 100)
values = f(domain, solution)

# Plot the values and set y-axis tick values
plt.figure()
plt.plot(domain, values, '-')
plt.yticks(range(2, 23, 2))

# Plot the known values for f(x) on the graph
plt.plot(matrix[0][1], b[0], 'or', markerfacecolor = 'none')
plt.plot(matrix[1][1], b[1], 'or', markerfacecolor = 'none')
plt.plot(matrix[2][1], b[2], 'or', markerfacecolor = 'none')
plt.plot(matrix[3][1], b[3], 'or', markerfacecolor = 'none')
plt.plot(matrix[4][1], b[4], 'or', markerfacecolor = 'none')

##############################################################################
# PART 3

# No values given for w by instructor so I randomly generated values
# until I found a suitable graph
w = [0.36148344289313444, -1.4351013014324692, -0.028050027659422827, 0.9028287877153592]

#######################################################################
# This section of code was given by the instructor
#######################################################################

# this defines the family of cubics
f = lambda w, x: w[0] + w[1] * x + w[2] * (x**2) + w[3] * (x**3)

# sample 20 equally spaced values between -2 and 2
dom = np.linspace(-2 , 2, 20)

# generate noisy data points, as values of a particular cubic + noise
val = f(w, dom) + np.random.randn(20) / 2

# generate ground truth samples for visualization only
val_truth = f(w, dom)

# plot
fig, ax = plt.subplots(1 , 1)
ax.plot(dom, val, 'bx')
ax.plot(dom, val_truth, '-g')
# ax.legend(('Noisy sampls', 'Ground truth'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Noisy samples and ground truth function')

#######################################################################

#######################################################################
# This section of code was written by me

# Create the matrix of x (dom) values (array: x^0, x^1, x^2, x^3)
matrix_a = []
for x in range(20):
    # This will hold each row of matrix_a with each pass
    holder = []
    for j in range(4):
        # Add each x raised to the j power to holder
        holder.append(pow(dom[x], j))
    # Add row to matrix_a
    matrix_a.append(holder)

# Use np.linalg.lstsq to compute least_sqr
least_sqr = np.linalg.lstsq(matrix_a, val, rcond=None)[0]

# Plot the approximated line and add the legend
plt.plot(dom, f(least_sqr, dom), '-r')
ax.legend(('Noisy samples', 'Ground truth', 'Learned model'))
