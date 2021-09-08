# basic libraries
import os
from copy import deepcopy
import subprocess

# import laspy
import laspy
from laspy import file

# import libraries for managing and plotting data
import numpy as np
import matplotlib.pyplot as plt
import math

# import sklearn libraries to sample numpy arrays
from sklearn.model_selection import train_test_split

# import data frame libraries
import pandas as pd

# import machine learning libraries
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

# import functions from external script
import "functions.py"


# test for GPU and tf version
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def main(traintype=['all'], shuffle=True, prefetch=False, cache=False):
    # import training files
    train_veg = v
    train_noveg = n

    #
