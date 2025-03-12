# basic libraries
import os, sys, time, datetime
from datetime import date

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
import matplotlib as mpl

# import libraries for managing and plotting data
import numpy as np

# import machine learning libraries
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

# import functions from other files
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__name__))), 'ML', 'src'))
from fileio import *
from tk_get_user_input import *
from vegindex import *
from miscfx import *
from modelbuilder import *

def train_reclass():
    print('test')