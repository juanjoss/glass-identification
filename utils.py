import random
import pandas as pd
import numpy as np
from constants import get_x_train_max, set_x_train_max, get_y_train_max, set_y_train_max

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

    # split in training and test data
    features_np_array = features.to_numpy()
    labels_np_array = labels.to_numpy()

    # training data (70%)
    x_train = features_np_array[:round(len(features_np_array)*0.7)]
    y_train = labels_np_array[:round(len(labels_np_array)*0.7)]

    # normalize training data
    set_x_train_max(np.amax(x_train))
    set_y_train_max(np.amax(y_train))

    x_train = x_train / get_x_train_max()
    y_train = y_train / get_y_train_max()

    # test data (30%)
    x_test = features_np_array[round(len(features_np_array)*0.7):]
    y_test = labels_np_array[round(len(labels_np_array)*0.7):]

    # normalize test data (with max and min from training data)
    x_test = x_test / get_x_train_max()
    y_test = y_test / get_y_train_max()

    x_train = [np.array([x]) for x in x_train]

    y_train_representation = []
    for i in range(len(y_train)):
        temp = np.zeros((1, 6))
        temp[0][int(y_train[i] * 6) - 1] = 1
        y_train_representation.append(temp)

    x_test = [np.array([x]) for x in x_test]

    y_test_representation = []
    for i in range(len(y_test)):
        temp = np.zeros((1, 6))
        temp[0][int(y_test[i] * 6) - 1] = 1
        y_test_representation.append(temp)

    return x_train, y_train_representation, x_test, y_test_representation

def shuffle(x: list, y: list):
    temp = list(zip(x, y))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return res1, res2

def labels_match(pred, correct) -> bool:
    label_pred = np.where(pred == np.amax(pred))[1][0] + 1
    label_correct = np.where(correct == 1)[1][0] + 1
    return label_pred == label_correct