# basic libraries
import os
import sys
from copy import deepcopy
import time
import subprocess

# import laspy
import laspy
if int(laspy.__version__.split('.')[0]) == 1:
    try:
        from laspy import file
    except Exception as e:
        sys.exit(e)

# import pdal (for manipulating and compressing LAS files)
# import pdal
# import json

# import libraries for managing and plotting data
import numpy as np
# import matplotlib.pyplot as plt
# import math
#
# # import sklearn libraries to sample numpy arrays
# from sklearn.model_selection import train_test_split
#
# # import scipy modules for KDTree fast spatial queries
# from scipy.spatial import cKDTree

# import data frame libraries
import pandas as pd

# import machine learning libraries
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

# import functions from other files
from src.fileio import *
from src.vegindex import *
from src.miscfx import *
from src.modelbuilder import *

# message box and file selection libraries
import tkinter
from tkinter import Tk
from tkinter.filedialog import askopenfile

class Args():
    ''' simple class to hold arguments '''
    pass
defs = Args()

# define default values/settings
# input files:
#   vegetation: input LAS or LAZ file containing only vegetation points
#   ground: input LAS or LAZ file containing only bare Earth points
defs.filein_vegetation = 'NA'
defs.filein_ground = 'NA'
# NOTE: If no filein_vegetation or filein_ground is/are specified, then the
# program will default to requesting the one or both files that are missing
# but required.



# for training:
#   epoch: number of training epochs
#   batch size: how many records should be aggregated (i.e. batched) together
#   prefetch: option to prefetch batches (may speed up training time)
#   shuffle: option to shuffle input data (good practice)
defs.training_epoch = 100
defs.training_batch_size = 1000
defs.training_prefetch = True
defs.training_shuffle = True

# for early stopping:
#   delta: The minmum change required to continue training beyond the number
#          of epochs specified by patience.
#   patience: The number of epochs to monitor change. If there is no improvement
#          greater than the value specified by delta, then training will stop.
defs.early_stop_patience = 5
defs.early_stop_delta = 0.001

def main(filevegetation, filebareearth, vegetationindices, trainingclassinbalancecorrection, trainingsplit, trainingdatareduction):
    # print info about TF and laspy packages
    print("Tensorflow Information:")
    # print tensorflow version
    print("   TF Version: {}".format(tf.__version__))
    print("   Eager mode: {}".format(tf.executing_eagerly()))
    print("   GPU name: {}".format(tf.config.experimental.list_physical_devices('GPU')))
    # list all available GPUs
    print("   Num GPUs Available: {}\n".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    print("laspy Information:")
    # print laspy version installed and configured
    print("   laspy Version: {}\n".format(laspy.__version__))

    # if no bare Earth or vegetation point clouds have been specified in the
    # user command line args, then request an input LAS/LAZ file for each
    if not filebareearth == 'NA':
        filebareearth = getfile(window_title='BARE EARTH point cloud')
    if not filevegetation == 'NA':
        filevegetation = getfile(window_title='VEGETATION point cloud')

    # the las2split() function performs the following actions:
    #   1) import training point cloud files
    #   2) compute vegetation indices
    #   3) split point clouds into training, testing, and validation sets
    train,test,val = las2split(filebareearth,
                           filevegetation,
                           veg_indices=vegetationindices,
                           class_imbalance_corr=trainingclassinbalancecorrection,
                           training_split=trainingsplit,
                           data_reduction=trainingdatareduction)

    # convert train, test, and validation to feature layers
    rgb_train_ds,train_ins,train_lyr = pd2fl(train, ['r','g','b','veglab'], shuf=defs.training_shuffle, ds_prefetch=defs.training_prefetch, batch_sz=defs.training_batch_size)
    rgb_val_ds,val_ins,val_lyr = pd2fl(val, ['r','g','b','veglab'], shuf=defs.training_shuffle, ds_prefetch=defs.training_prefetch, batch_sz=defs.training_batch_size)
    rgb_test_ds,test_ins,test_lyr = pd2fl(test, ['r','g','b','veglab'], shuf=defs.training_shuffle, ds_prefetch=defs.training_prefetch, batch_sz=defs.training_batch_size)

    #

if __name__ == '__main__':
    # get default args from the defs class
    # (these will be updated as necessary)
    arg_vegetation_file = defs.filein_vegetation
    arg_ground_file = defs.filein_ground
    arg_training_epoch = defs.training_epoch
    arg_training_batch_size = defs.training_batch_size
    arg_training_prefetch = defs.training_prefetch
    arg_training_shuffle = defs.training_shuffle

    argv = sys.argv[1:]
    try:
        opts,args = getopt.getopt(argv,"v:g:n:")
    except Exception as e:
        print(e)
        sys.exit()
    main()
