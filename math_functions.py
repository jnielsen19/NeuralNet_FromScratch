import numpy as np
from process import m, one_hot_encode

#INIT ACTIVATION FUNCTIONS
#   Rectified Linear Unit Fn(x=x > 0; x=0 <= 0) | (see math notes for further information)
def ReLU(input):
    return np.maximum(input, 0) # element wise operation, that gets the maximum between 0 and each element in input bc input will be an array

def Softmax(input):
    A = np.exp(input) / sum(np.exp(input))
    return A
    # these again are element wise operations which take each element of input | for more info see math notes
    # worth noting the way np.sum works: preserves the colomns the input/output columns still correspond, 
    #   however, collapses rows of dataframe to the sum of every row in col

def ReLU_deriv(input):
    return input > 0 #if you think about the way ReLU works combined with when Booleans are converted to numbers(True=1, False=0)
    # the slope of the ReLU fn is either 0 or 1 therefore deriv = 1 only when input is 1/True which is whats done above

#BEGIN TRAIN FUNCTIONS

def forward_propagation(w1, b1, w2, b2, I0):
    #This function follows notation and steps of math in notes 
    Z_1 = w1.dot(I0) + b1
    L_1 = ReLU(Z_1)
    Z_2 = w2.dot(L_1) + b2
    L_2 = Softmax(Z_2)

    return Z_1, L_1, Z_2, L_2

def backward_propagation(Z_1, L_1, Z_2, L_2, w1, w2, I0, lmbda):
    one_hot_lambda = one_hot_encode(lmbda)
    dZ_2 = L_2 - one_hot_lambda
    dw_2 = (1/m) * dZ_2.dot(L_1.T)
    db_2 = (1/m) * np.sum(dZ_2)
    dZ_1 = w2.T.dot(dZ_2) * ReLU_deriv(Z_1)
    dw_1 = (1/m) * dZ_1.dot(I0.T)
    db_1 = (1/m) * np.sum(dZ_1)

    return dw_1, db_1, dw_2, db_2

def update_parameters(alpha, w1, b1, w2, b2, dw_1, db_1, dw_2, db_2):
    w1 = w1 - alpha*dw_1
    w2 = w2 - alpha*dw_2
    b1 = b1 - alpha*db_1
    b2 = b2 - alpha*db_2

    return w1, b1, w2, b2