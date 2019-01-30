# Simple two layer neural network
# Three input neurons and single hidden layer leading to single output
# ---------------------
# | Inputs  | Outputs |
# ---------------------
# | 0,0,1   | 0       |
# | 0,1,1   | 0       |
# | 1,0,1   | 1       |
# | 1,1,1   | 1       |
# ---------------------
# Note the correspondence between first neuron and output is 100%

# Understand the variables and the operations between them
#   X	    Input dataset matrix where each row is a training example
#   y	    Output dataset matrix where each row is a training example
#   l0	    First Layer of the Network, specified by the input data
#   l1	    Second Layer of the Network, otherwise known as the hidden layer
#   syn0	First layer of weights, Synapse 0, connecting l0 to l1.
#  x.dot(y)	If x and y are vectors, this is a dot product.
#               If both are matrices, it's a matrix-matrix multiplication.
#               If only one is a matrix, then it's vector matrix multiplication.

import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)
