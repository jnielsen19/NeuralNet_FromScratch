from process import m, train_labels, train_values
from model_functions import gradient_descent, test_model


w1, b1, w2, b2 = gradient_descent(train_values, train_labels, 500, 0.1)


test_model(27, w1, b1, w2, b2, train_values, train_labels)