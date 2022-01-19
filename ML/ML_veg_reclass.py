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

# import libraries for managing and plotting data
import numpy as np

# import machine learning libraries
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

# import functions from other files
from src.fileio import *
from src.tk_get_user_input_RECLASS_ONLY import *
from src.vegindex import *
from src.miscfx import *
from src.modelbuilder import *


def main(default_values, verbose=True):
    # parse any command line arguments (if present)
    default_values.parse_cmd_arguments()

    if default_values.gui:
        request_window = App()
        request_window.create_widgets(default_values)
        request_window.mainloop()

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

    # list of acceptable geometric metrics to calculate
    # this list is subject to expansion or reduction depending on new code
    acceptablegeometrics = ['sd']

    # get today's date as string
    tdate = str(date.today()).replace('-','')

    ## RECLASSIFY A FilE USING TRAINED MODEL FILE
    # get the input model
    # while default_values.model_file == 'NA':
    #     default_values.model_file = getfile(window_title='Select trained model h5 file')
    # load the model
    try:
        reclassmodel = tf.keras.models.load_model(default_values.model_file)
        globals()[reclassmodel.name] = reclassmodel
    except Exception as e:
        sys.exit(e)

    # get the model name from the loaded file
    if default_values.model_name == 'NA':
        try:
            default_values.model_name = reclassmodel.name
        except Exception as e:
            print(e)

    ''' LOOK AT COMBINING MODEL_NAME WITH MODEL_FILE'''

    # get model inputs from the loaded file
    default_values.model_inputs = [f.name for f in reclassmodel.inputs]
    if verbose:
        print(default_values.model_inputs)

    # check if any geometry metrics are specified
    geomet = list(set(acceptablegeometrics).intersection(default_values.model_inputs))

    # print the model summary
    if verbose:
        print(reclassmodel.summary())

    # # get the input dense cloud
    # if default_values.reclassfile == 'NA':
    #     default_values.reclassfile = getfile(window_title='Select point cloud to reclassify')

    # if the reclass model is already a list of models, then proceed
    if isinstance(reclassmodel, list):
        predict_reclass_write(default_values.reclassfile,
                                reclassmodel,
                                threshold_vals=default_values.reclass_thresholds,
                                batch_sz=default_values.training_batch_size,
                                ds_cache=default_values.training_cache,
                                geo_metrics=geomet,
                                geom_rad=default_values.geometry_radius)
    # else if the reclass model is not a list of models, then convert it to a
    # list for use in the predict_reclass_write() function
    elif not isinstance(reclassmodel, list):
        try:
            predict_reclass_write(default_values.reclassfile,
                                    [reclassmodel],
                                    threshold_vals=default_values.reclass_thresholds,
                                    batch_sz=default_values.training_batch_size,
                                    ds_cache=default_values.training_cache,
                                    geo_metrics=geomet,
                                    geom_rad=default_values.geometry_radius)
        except Exception as e:
            print('ERROR: Unable to reclassify the input file. See below for specific error:')
            sys.exit(e)

    # get the model inputs from the loaded file
if __name__ == '__main__':
    defs = Args('defs')

    main(default_values=defs)
