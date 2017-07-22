# -*- coding: utf-8 -*-

from __future__ import print_function

import sklearn
import sklearn.model_selection
from keras.datasets import mnist

VAL_SEED = 123456


def load_mnist():
    (x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()

    X_data = x_train0.reshape( ( x_train0.shape[0], x_train0.shape[1]*x_train0.shape[1] ) )
    y_data = y_train0

    X_test = x_test0.reshape( ( x_test0.shape[0], x_test0.shape[1]*x_test0.shape[2] ) )

    x1, x2, y1, y2 = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.2, random_state=VAL_SEED)

    return x1, y1, x2, y2, X_test, y_test0
