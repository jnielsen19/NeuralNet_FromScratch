import numpy as np
from matplotlib import pyplot as plt
from math_functions import forward_propagation, backward_propagation, update_parameters
from process import init_params


def gradient_descent(I0, lmbda, iter, alpha):
    w1, b1, w2, b2 = init_params()

    for i in range(iter):
        Z_1, L_1, Z_2, L_2 = forward_propagation(w1, b1, w2, b2, I0)
        dw_1, db_1, dw_2, db_2 = backward_propagation(Z_1, L_1, Z_2, L_2, w1, w2, I0, lmbda)
        w1, b1, w2, b2 = update_parameters(alpha, w1, b1, w2, b2, dw_1, db_1, dw_2, db_2)
        # every 10 itertions will print train statistics
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(L_2)
            print(get_accuracy(predictions, lmbda))

    return w1, b1, w2, b2


def get_predictions(L_2):
    #gets the index of greatest probability from the 10 nodes in the output layer (2nd layer) 
    #   -> essentially equivilent to the digit prediction of the neural net(0-9)
    return np.argmax(L_2, 0) 

def get_accuracy(predictions, lmbda):
    #returns the amount of predictions that are correct divided by the total number of images
    print(predictions, lmbda)
    return np.sum(predictions == lmbda) / lmbda.size 

def predict(values, w1, b1, w2, b2):
    _,_,_, L_2 = forward_propagation(w1, b1, w2, b2, values)
    predictions = get_predictions(L_2)
    return predictions


def test_model(pos, w1, b1, w2, b2, train_values, train_labels):
    image_at_pos = train_values[:, pos, None]
    model_prediction = predict(image_at_pos, w1, b1, w2, b2)
    value_int = train_labels[pos]

    print('Prediction: ', model_prediction)
    print('Actual Value: ', value_int)

    image_at_pos = image_at_pos.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image_at_pos, interpolation='nearest')
    plt.show()
