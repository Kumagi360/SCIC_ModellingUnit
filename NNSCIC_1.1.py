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



# this, numpy - a linear algebra library - is the only dependency we need.
import numpy as np

# this is our 'nonlinearity' - sigmoid function maps any value to a value between 0 and 1.
# this converts numbers into probabilties
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
# each row is a single 'training example' - 4 examples with 3 nodes each
# each column corresponds to one input node
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
# .T is a transpose function (good practice)
# tranpose converts 4 column 1 row matrix into 1 column 4 rows
# thus we have four training exmaples each with 3 inputs and one output label
y = np.array([[0, 0, 1, 1]]).T

# 'seed' random numbers to vary weights in teh same way across every training
# just a good practice (deterministic)
np.random.seed(1)

# initialize weights randomly with mean 0 - often producs best results
# syn0 implies synapse zero
# weight matrix is of dimension (3,1) as 3 inputs and 1 output (hence three connections)
# this is the heart of the learning as all learing in this NN is changes to the weights here
syn0 = 2 * np.random.random((3, 1)) - 1


# this loop trains our network iterating a number of times over training code
for iter in range(10000):
    # forward propagation

    # l0 is just out first layer - the inputs of four training examples
    # this is known as 'full batch' training where all the examples are treated as one
    l0 = X



    # guess stage - tries to predict the output given the input and current weights
    # first numpy multiplies l0 and syn0 (dot product as they are matrices)
    # dot product - output is matrix of number of rows of first matrix and columns of second
    # then passes this to sigmoid function
    l1 = nonlin(np.dot(l0, syn0))
    print("l1")
    print(l1)


    # how much did we miss?
    # comes up with vector of postive and negative reflecting how much we missed
    l1_error = y - l1
    print("l1_error")
    print(l1_error)

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    # nonlin(l1, True) returns the derivative of each point at l1
        # the higher the slope the more unconfident and the higher the weight change
    # thus, either a high error or a high derivative both will cause large adjustment
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    # this allows the weight to be shifted depending on both delta and the input value
    syn0 += np.dot(l0.T, l1_delta)
    print("syn0")
    print(syn0)


print("Output After Training:")
print(l1)
