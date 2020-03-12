import numpy as np


def xor_3_input():
    """ Truth table for a XOR operation with 3 inputs. """
    X = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])  # input

    Y = np.array([[0.],
                  [1.],
                  [1.],
                  [0.],
                  [1.],
                  [0.],
                  [0.],
                  [1.]])  # output (true if odd number of inputs is true)

    X = X.T
    Y = Y.T

    return X, Y
