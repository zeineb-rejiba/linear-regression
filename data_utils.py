import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


def load_training_data(data_path, col_names):
    """
    loads the training data from the specified path
    :param data_path: training data path
    :param col_names: the column names
    :return: a dataframe object, the features array and the target values
    """
    df = pd.read_csv(data_path, names=col_names, sep=',', dtype=np.float64, header=None)
    cols = df.columns
    X = df[cols[:-1]]  # define the features as the set of all columns of the dataframe except the last one
    y = df[cols[-1]]  # the last column of the dataframe corresponds to the target values
    return df, X, y


def prepare_data(X, y):
    """
    prepares the data by adding a column of 1s to account for the intercept term. Then, it converts the X and y to tensorflow tensors.
    :param X: the training data features
    :param y: the target values
    :return: the X and y values after conversion
    """
    # insert a column of 1s to the left of X. This is to account for the intercept term.
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # convert X and y to tensorflow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y.to_numpy(), dtype=tf.float32)

    # I had to specifically force the following reshape, otherwise the dimensions were misinterpreted
    y = tf.reshape(y, [y.shape[0],1])

    return X, y