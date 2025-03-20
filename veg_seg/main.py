# main.py
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

import matplotlib as mpl

# import libraries for managing and plotting data
import numpy as np

# import machine learning libraries
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *

# import functions from other files
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__name__))), 'ML'))
from .ML_vegfilter import *
from .ML_veg_reclass import *
from .ML_veg_train import *
# from src.fileio import *
from .src.tk_get_user_input import *
from .src.tk_get_user_input_TRAIN_ONLY import *
from .src.tk_get_user_input_RECLASS_ONLY import *
# from src.vegindex import *
# from src.miscfx import *
# from src.modelbuilder import *

# def main():
#     print('testing')
def veg_filter():
    ml_veg_filter(default_values=Args('defs'))

def veg_reclass():
    ml_veg_reclass(default_values=Args_reclass_only('defs'))

def veg_train():
    ml_veg_train(default_values=Args_train_only('defs'))