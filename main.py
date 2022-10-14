from ftplib import error_reply
import pandas as pd
import numpy as np
from network import Network
from layer import FCLayer, ActivationLayer
from activation import sigmoid, sigmoid_prime, relu, relu_prime
from loss import MSE, MSE_prime

def read_data():
    """
        Function used to read and load glass data.
    """
    # read data
    data = pd.read_csv("./data/glass.data")
    data = data.drop(data.columns[[0]], axis = 1) # delete ID column
    data = data.sample(frac=1).reset_index(drop=True) # shuffle data

    # extract features and labels
    features = data.iloc[::, 0:9]
    labels = data["glass_type"]

    # normalize feature variables
    normalized_features = (features - features.min()) / (features.max() - features.min())

    # split in training and test data
    features_np_array = normalized_features.to_numpy()
    labels_np_array = labels.to_numpy()

    x_train = features_np_array[:round(len(features_np_array)*0.7)] # 70% train
    x_test = features_np_array[round(len(features_np_array)*0.7):] # 30% test

    y_train = labels_np_array[:round(len(labels_np_array)*0.7)]
    y_test = labels_np_array[round(len(labels_np_array)*0.7):]

    x_train = [np.array([x]) for x in x_train]
    y_train = [np.array([x]) for x in y_train]
    x_test = [np.array([x]) for x in x_test]
    y_test = [np.array([x]) for x in y_test]

    return x_train, y_train, x_test, y_test

def run():
    x_train, y_train, x_test, y_test = read_data()

    # create and define the network
    network = Network()
    network.layer(layer=FCLayer(9, 7))
    network.layer(layer=ActivationLayer(sigmoid, sigmoid_prime))
    network.layer(layer=FCLayer(7, 1))
    network.layer(layer=ActivationLayer(relu, relu_prime))

    # train
    network.use_loss(MSE, MSE_prime)
    network.train(x_train, y_train, 1000, 0.01)

    # test
    out = network.predict(x_test)

    count = 0
    for i in range(len(out) - 1):
        predicted = np.ceil(out[i][0])
        correct = y_test[i]

        if predicted == correct:
            count += 1
    
    print("test_success_rate: ", count / len(out))


if __name__ == "__main__":
    run()