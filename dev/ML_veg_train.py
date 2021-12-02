# basic libraries
import os, sys, time, getopt

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
from src.vegindex import *
from src.miscfx import *
from src.modelbuilder import *

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

# model name:
defs.model_output_name = ''

# model inputs and vegetation indices of interest:
defs.model_inputs = ['r','g','b']
defs.model_vegetation_indices = 'rgb'
defs.model_include_coords = False
defs.model_include_geom = False
defs.model_nodes = [16,16,16]
defs.model_dropout = 0.2

# for training:
#   epoch: number of training epochs
#   batch size: how many records should be aggregated (i.e. batched) together
#   prefetch: option to prefetch batches (may speed up training time)
#   shuffle: option to shuffle input data (good practice)
#   training split: proportion of the data to use for training (remainder will
#                   be used for testing of the model performance)
defs.training_epoch = 100
defs.training_batch_size = 1000
defs.training_prefetch = True
defs.training_shuffle = True
defs.training_split = 0.7
defs.training_class_imbalance_corr = True
defs.training_data_reduction = 1.0

# for early stopping:
#   delta: The minmum change required to continue training beyond the number
#          of epochs specified by patience.
#   patience: The number of epochs to monitor change. If there is no improvement
#          greater than the value specified by delta, then training will stop.
defs.model_early_stop_patience = 5
defs.model_early_stop_delta = 0.001

def main(default_values,
            verbose=True):
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
    if default_values.filein_ground == 'NA':
        default_values.filein_ground = getfile(window_title='BARE EARTH point cloud')
    if default_values.filein_vegetation == 'NA':
        default_values.filein_vegetation = getfile(window_title='VEGETATION point cloud')
    if default_values.model_output_name == '':
        default_values.model_output_name = getmodelname()

    # the las2split() function performs the following actions:
    #   1) import training point cloud files
    #   2) compute vegetation indices
    #   3) split point clouds into training, testing, and validation sets
    if 'sd' in default_values.model_inputs:
        train,test,val = las2split(default_values.filein_ground,
                               default_values.filein_vegetation,
                               veg_indices=default_values.model_vegetation_indices,
                               class_imbalance_corr=default_values.training_class_imbalance_corr,
                               training_split=default_values.training_split,
                               data_reduction=default_values.training_data_reduction,
                               geom_metrics='sd')
    else:
        train,test,val = las2split(default_values.filein_ground,
                               default_values.filein_vegetation,
                               veg_indices=default_values.model_vegetation_indices,
                               class_imbalance_corr=default_values.training_class_imbalance_corr,
                               training_split=default_values.training_split,
                               data_reduction=default_values.training_data_reduction)

    # append columns/variables of interest list
    variablesofinterest.append('veglab')

    # convert train, test, and validation to feature layers
    train_ds,train_ins,train_lyr = pd2fl(train, default_values.model_inputs,
                                            shuf=default_values.training_shuffle,
                                            ds_prefetch=default_values.training_prefetch,
                                            batch_sz=default_values.training_batch_size,
                                            ds_cache=default_values.training_cache)
    val_ds,val_ins,val_lyr = pd2fl(val, default_values.model_inputs,
                                            shuf=default_values.training_shuffle,
                                            ds_prefetch=default_values.training_prefetch,
                                            batch_sz=default_values.training_batch_size,
                                            ds_cache=default_values.training_cache)
    test_ds,test_ins,test_lyr = pd2fl(test, default_values.model_inputs,
                                            shuf=default_values.training_shuffle,
                                            ds_prefetch=default_values.training_prefetch,
                                            batch_sz=default_values.training_batch_size,
                                            ds_cache=default_values.training_cache)

    # print model input attributes
    if verbose:
        print("Model inputs:\n   {}".format(default_values.model_inputs))

    # build and train model
    mod = build_model(model_name=default_values.model_output_name,
                        model_inputs=default_values.model_inputs,
                        input_feature_layer=train_lyr,
                        training_tf_dataset=train_ds,
                        validation_tf_dataset=val_ds,
                        nodes=default_values.model_nodes,
                        activation_fx='relu',
                        dropout_rate=default_values.model_dropout,
                        loss_metric='mean_squared_error',
                        model_optimizer='adam',
                        earlystopping=[default_values.model_early_stop_patience,default_values.model_early_stop_delta],
                        dotrain=True,
                        dotrain_epochs=default_values.training_epoch,
                        verbose=True)

    # # evaluate the model
    # loss,accuracy = mod.evaluate(test_ds, verbose=1)
    # # write to model metadata file
    # # 1) model name,
    # # 2) model inputs (file names),
    # # 3) model inputs (variables),
    # # 4) creation timestamp,
    # # 5) model accuracy, and
    # # 6) model summary

    # # check if saved model dir already exists (create if not present)
    # if not os.path.isdir('saved_models'):
    #     os.makedirs('saved_models')
    #
    # # save the complete model
    # mod.save(ps.path.join('saved_models',output_model_name))
    # # save the model weights as H5 file
    # mod.save(ps.path.join('saved_models',(output_model_name+'.h5')))
    #
    # # print model summary to console
    # if verbose:
    #     print(mod.summary())

if __name__ == '__main__':
    arg_model_early_stop_patience = defs.model_early_stop_patience
    arg_model_early_stop_delta = defs.model_early_stop_delta
    arg_training_epoch = defs.training_epoch
    arg_training_batch_size = defs.training_batch_size
    arg_training_prefetch = defs.training_prefetch
    arg_training_shuffle = defs.training_shuffle

    # parse command line arguments
    argv = sys.argv[1:]
    try:
        opts,args = getopt.getopt(argv,"v:g:m:vi:mi:mn:md:te:tb")
    except Exception as e:
        print(e)
        sys.exit()
    for opt,arg in opts:
        if opt in ['-v','--vegfile']:
            # input vegetation only dense cloud/point cloud
            defs.filein_vegetation = str(arg)
        elif opt in ['-g','--groundfile']:
            # input bare-Earth only dense cloud/point cloud
            defs.filein_ground = str(arg)
        elif opt in ['-m','--modelname']:
            # model output name (used to save the model)
            defs.model_output_name = str(arg)
        elif opt in ['-vi','--vegind']:
            # because the input argument is handled as a single string, we need
            # to strip the brackets, split by the delimeter, and then re-form it
            # as a list of characters/strings
            defs.model_vegetation_indices = str(arg).strip('[').strip(']').split(',')
        elif opt in ['-mi','--modelinputs']:
            # because the input argument is handled as a single string, we need
            # to strip the brackets, split by the delimeter, and then re-form it
            # as a list of characters/strings
            defs.model_inputs = str(arg).strip('[').strip(']').split(',')
        elif opt in ['-mn','--modelnodes']:
            # because the input argument is handled as a string, we need to
            # strip the brackets and split by the delimeter, convert each string
            # to an integer, and then re-map the converted integers to a list
            defs.model_nodes = list(map(int, str(arg).strip('[').strip(']').split(',')))
        elif opt in ['-md','--modeldropout']:
            # model dropout value must be within 0.0 and 1.0
            if 0.0 > arg and arg < 1.0:
                defs.model_dropout = arg
            else:
                print('Invalid dropout specified, using default probability of 0.2')
        elif opt in ['-mes','--modelearlystop']:
            # option to define early stopping criteria
            earlystopcriteria = list(map(float, str(arg).strip('[').strip(']').split(',')))
            defs.model_early_stop_patience = earlystopcriteria[0]
            defs.model_early_stop_delta = earlystopcriteria[1]
        elif opt in ['-te','--epochs']:
            # define training epochs
            defs.training_epoch = arg
        elif opt in ['-tb','--batchsize']:
            # training batch size
            defs.training_batch_size
    main(default_values=defs)
