import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
import os
import sys
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy import empty, arange, average, minimum, maximum
from numpy.random import RandomState
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.ops import math_ops
import scipy.stats


def get_yu_model(input_shape, init_bias=0):
    def bce_loss(y_true, y_pred):
        bce = y_true * math_ops.log(y_pred + epsilon())
        bce += (1 - y_true) * math_ops.log(1 - y_pred + epsilon())
        return -bce

    FP_FACTOR = 1000 # 2000: 2 fp, 3000: didn't work

    def tn_loss(y_true, y_pred): return (1 - y_true) * (1 - y_pred) * bce_loss(y_pred, y_true)
    def fp_loss(y_true, y_pred): return (1 - y_true) * (    y_pred) * bce_loss(y_pred, y_true) * FP_FACTOR
    def fn_loss(y_true, y_pred): return (    y_true) * (1 - y_pred) * bce_loss(y_pred, y_true)
    def tp_loss(y_true, y_pred): return (    y_true) * (    y_pred) * bce_loss(y_pred, y_true)

    def fbeta_bce_loss(y_true, y_pred, beta = 2):
        beta_sq = beta ** 2

        bce = bce_loss(y_pred, y_true)

        #### prediction       false                      true
        # actual
        # false            true negative           false positive
        # true         false negative (miss)      true positive (hit)

        tn, fp = (1 - y_true) * (1 - y_pred) * bce, (1 - y_true) * (    y_pred) * bce * FP_FACTOR
        fn, tp = (    y_true) * (1 - y_pred) * bce, (    y_true) * (    y_pred) * bce

        return tn + fp + fn + tp

        return - (1 + beta_sq) * tp_loss / ((beta_sq * y_true) + tp_loss + fp_loss)

    tf.random.set_seed(2)

    # build network
    yu_clf = tf.keras.Sequential()
    yu_clf.add(tf.keras.layers.Dense(
        10, activation='relu',
        input_shape=input_shape, 
        kernel_initializer='random_normal'
    ))
    yu_clf.add(tf.keras.layers.Dense(10, activation='relu'))
    yu_clf.add(tf.keras.layers.Dense(20, activation='relu'))
    yu_clf.add(tf.keras.layers.Dense(10, activation='relu'))
    yu_clf.add(tf.keras.layers.Dense(
        1, activation='sigmoid',
        bias_initializer=tf.keras.initializers.Constant(init_bias),
    ))
    opt = tf.keras.optimizers.Adam(learning_rate=.001)
    yu_clf.compile(
        optimizer=opt, 
        loss=fbeta_bce_loss, 
        metrics=[
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve    
            tn_loss, fp_loss, fn_loss, tp_loss,
        ],
    )
    return yu_clf

