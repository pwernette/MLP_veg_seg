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
from src.argument_parsing import *
from src.vegindex import *
from src.miscfx import *
from src.modelbuilder import *

class Args():
    ''' simple class to hold arguments '''
    pass
defs = Args()

# define default values/settings
# input files:
#   vegetation: input LAS/LAZ file containing only vegetation points
#   ground: input LAS/LAZ file containing only bare Earth points
#   reclassfile: input LAS/LAZ file to reclassify using the specified model
defs.filein_vegetation = 'NA'
defs.filein_ground = 'NA'
defs.reclassfile = 'NA'
# NOTE: If no filein_vegetation or filein_ground is/are specified, then the
# program will default to requesting the one or both files that are missing
# but required.

# model file used for reclassification
defs.model_file = 'NA'

# model name:
defs.model_name = 'NA'

# model inputs and vegetation indices of interest:
#   model_inputs: list of input variables for model training and prediction
#   model_vegetation_indices: list of vegetation indices to calculate
#   model_nodes: list with the number of nodes per layer
#      NOTE: The number of layers corresponds to the list length. For example:
#          8,8,8 --> 3 layer model with 8 nodes per layer
#          8,16 --> 2 layer model with 8 nodes (L1) and 16 nodes (L2)
#   model_dropout: probability of dropping out (i.e. not using) any node
#   geometry_radius: 3D radius used to compute geometry information over
defs.model_inputs = ['r','g','b']
defs.model_vegetation_indices = 'rgb'
defs.model_nodes = [16,16,16]
defs.model_dropout = 0.2
defs.geometry_radius = 0.10

# for early stopping:
#   delta: The minmum change required to continue training beyond the number
#          of epochs specified by patience.
#   patience: The number of epochs to monitor change. If there is no improvement
#          greater than the value specified by delta, then training will stop.
defs.model_early_stop_patience = 5
defs.model_early_stop_delta = 0.001

# for training:
#   epoch: number of training epochs
#   batch size: how many records should be aggregated (i.e. batched) together
#   prefetch: option to prefetch batches (may speed up training time)
#   cache: option to cache batches ahead of time (speeds up training time)
#   shuffle: option to shuffle input data (good practice)
#   training split: proportion of the data to use for training (remainder will
#                   be used for testing of the model performance)
#   class imbalance corr: option to correct for class size imbalance
#   data reduction: setting this to a number between 0 and 1 will reduce the
#                   overall volume of data used for training and validation
defs.training_epoch = 100
defs.training_batch_size = 1000
defs.training_cache = True
defs.training_prefetch = True
defs.training_shuffle = True
defs.training_split = 0.7
defs.training_class_imbalance_corr = True
defs.training_data_reduction = 1.0

# for plotting:
#   plotdir: plotting direction (horizontal (h) or vertical (v))
defs.plotdir = 'v'

# for reclassification
#   reclass_thresholds: list of thresholds used for reclassification
defs.reclass_thresholds = [0.6]

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

    # list of acceptable geometric metrics to calculate
    # this list is subject to expansion or reduction depending on new code
    acceptablegeometrics = ['sd']

    # parse any command line arguments (if present)
    parse_cmd_arguments(default_values)

    # # if no bare Earth or vegetation point clouds have been specified in the
    # # user command line args, then request an input LAS/LAZ file for each
    # if default_values.filein_ground == 'NA':
    #     default_values.filein_ground = getfile(window_title='Specitfy input BARE-EARTH (GROUND) point cloud')
    # if default_values.filein_vegetation == 'NA':
    #     default_values.filein_vegetation = getfile(window_title='Specify input VEGETATION point cloud')
    # while default_values.model_output_name == 'NA':
    #     default_values.model_output_name = getmodelname()
    #
    # # the las2split() function performs the following actions:
    # #   1) import training point cloud files
    # #   2) compute vegetation indices
    # #   3) split point clouds into training, testing, and validation sets
    # if 'sd' in default_values.model_inputs:
    #     train,test,val = las2split(default_values.filein_ground,
    #                            default_values.filein_vegetation,
    #                            veg_indices=default_values.model_vegetation_indices,
    #                            class_imbalance_corr=default_values.training_class_imbalance_corr,
    #                            training_split=default_values.training_split,
    #                            data_reduction=default_values.training_data_reduction,
    #                            geom_metrics='sd')
    # else:
    #     train,test,val = las2split(default_values.filein_ground,
    #                            default_values.filein_vegetation,
    #                            veg_indices=default_values.model_vegetation_indices,
    #                            class_imbalance_corr=default_values.training_class_imbalance_corr,
    #                            training_split=default_values.training_split,
    #                            data_reduction=default_values.training_data_reduction)
    #
    # # append columns/variables of interest list
    # default_values.model_inputs.append('veglab')
    #
    # # convert train, test, and validation to feature layers
    # train_ds,train_ins,train_lyr = pd2fl(train, default_values.model_inputs,
    #                                         shuf=default_values.training_shuffle,
    #                                         ds_prefetch=default_values.training_prefetch,
    #                                         batch_sz=default_values.training_batch_size,
    #                                         ds_cache=default_values.training_cache)
    # val_ds,val_ins,val_lyr = pd2fl(val, default_values.model_inputs,
    #                                         shuf=default_values.training_shuffle,
    #                                         ds_prefetch=default_values.training_prefetch,
    #                                         batch_sz=default_values.training_batch_size,
    #                                         ds_cache=default_values.training_cache)
    # test_ds,test_ins,test_lyr = pd2fl(test, default_values.model_inputs,
    #                                         shuf=default_values.training_shuffle,
    #                                         ds_prefetch=default_values.training_prefetch,
    #                                         batch_sz=default_values.training_batch_size,
    #                                         ds_cache=default_values.training_cache)
    #
    # # print model input attributes
    # if verbose:
    #     print("\nModel inputs:\n   {}".format(list(train_ins)))
    #
    # # build and train model
    # mod,tt = build_model(model_name=default_values.model_output_name,
    #                     model_inputs=train_ins,
    #                     input_feature_layer=train_lyr,
    #                     training_tf_dataset=train_ds,
    #                     validation_tf_dataset=val_ds,
    #                     nodes=default_values.model_nodes,
    #                     activation_fx='relu',
    #                     dropout_rate=default_values.model_dropout,
    #                     loss_metric='mean_squared_error',
    #                     model_optimizer='adam',
    #                     earlystopping=[default_values.model_early_stop_patience,default_values.model_early_stop_delta],
    #                     dotrain=True,
    #                     dotrain_epochs=default_values.training_epoch,
    #                     verbose=True)
    #
    # # evaluate the model
    # print('\nEvaluating model with validation set...')
    # model_loss,model_accuracy = mod.evaluate(test_ds, verbose=2)
    #
    # get today's date as string
    tdate = str(date.today()).replace('-','')
    #
    # # check if saved model dir already exists (create if not present)
    # if not os.path.isdir('saved_models_'+tdate):
    #     os.makedirs('saved_models_'+tdate)
    #
    # print('\nWriting summary log file...')
    # # write model information and metadata to output txt file
    # with open(os.path.join('saved_models_'+tdate,'SUMMARY_'+default_values.model_output_name+'.txt'),'w') as fh:
    #     # print model summary to output file
    #     # Pass the file handle in as a lambda function to make it callable
    #     mod.summary(print_fn=lambda x: fh.write(x+'\n'))
    #     fh.write('created: {}\n'.format(tdate))
    #     fh.write('bare-Earth file: {}\n'.format(default_values.filein_ground))
    #     fh.write('vegetation file: {}\n'.format(default_values.filein_vegetation))
    #     fh.write('model inputs: {}\n'.format(list(default_values.model_inputs)))
    #     fh.write('validation accuracy: {}\n'.format(model_accuracy))
    #     fh.write('validation loss: {}\n'.format(model_loss))
    #     fh.write('train time: {}'.format(datetime.timedelta(seconds=tt))+'\n')
    #
    # print('\nSaving model as .h5 and sub-directory in {}...'.format('saved_models_'+tdate))
    # # save the complete model (will create a new folder with the saved model)
    # mod.save(os.path.join('saved_models_'+tdate,default_values.model_output_name))
    # # save the model weights as H5 file
    # mod.save(os.path.join('saved_models_'+tdate,(default_values.model_output_name+'.h5')))
    #
    # print('\nPlotting model...')
    # # plot the model as a PNG
    # plot_model(mod, to_file=(os.path.join('saved_models_'+tdate,str(default_values.model_output_name)+'_GRAPH.png')), rankdir=default_values.plotdir, dpi=300)

    ## RECLASSIFY A FilE USING TRAINED MODEL FILE
    # get the input model
    while default_values.model_file == 'NA':
        default_values.model_file = getfile(window_title='Select trained model h5 file')
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

    # get the input dense cloud
    if default_values.reclassfile == 'NA':
        default_values.reclassfile = getfile(window_title='Select point cloud to reclassify')

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
    main(default_values=defs)
