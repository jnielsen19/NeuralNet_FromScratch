import pandas as pd
import numpy as np

#INIT PARAMETERS
def init_params():
    #NOTE: Dimensions for all arrays are shown in math notebook
    w1 = np.random.rand(10, 784) - 0.5 # rand values distributed between 0 & 1 
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2

#ONE-HOT-ENCODING
def one_hot_encode(labels):
    one_hot_Y = np.zeros((labels.size, labels.max() + 1))
    one_hot_Y[np.arange(labels.size), labels] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y



dataset = pd.read_csv('data/train/train.csv')

#convert to np arrary
dataframe = np.array(dataset)

#getting the size of data frame | m = # of rows & n = # of cols + 1 becaus of labels column
m, n = dataframe.shape

#elimate the possiblity that the neural net learns some invalid patterns based on the downloaded presentation of the data by randomizing order
np.random.shuffle(dataframe)

#BELOW: Preprocessing Step - pulling out labels and creating matrix of values

#pulling out some of the data to prevent overfitting
dev_data = dataframe[0:1000].T
dev_labels = dev_data[0]
dev_values = dev_data[1:n]
dev_values = dev_values / 255

# building training set
train_data = dataframe[1000:m].T
train_labels = train_data[0]
train_values = train_data[1:n]
train_values = train_values / 255