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
from src.tk_get_user_input import *
from src.vegindex import *
from src.miscfx import *
from src.modelbuilder import *

def main(default_values, verbose=True):
    # list of acceptable geometric metrics to calculate
    # this list is subject to expansion or reduction depending on new code
    acceptablegeometrics = ['sd']

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

    # parse any command line arguments (if present)
    # parse_cmd_arguments(default_values)

    # if no bare Earth or vegetation point clouds have been specified in the
    # user command line args, then request an input LAS/LAZ file for each
    # if default_values.filein_ground == 'NA':
    #     default_values.filein_ground = getfile(window_title='Specitfy input BARE-EARTH (GROUND) point cloud')
    # if default_values.filein_vegetation == 'NA':
    #     default_values.filein_vegetation = getfile(window_title='Specify input VEGETATION point cloud')
    # if default_values.model_output_name == 'NA':
    #     default_values.model_output_name = getmodelname()
    # # get the input dense cloud
    # if default_values.reclassfile == 'NA':
    #     default_values.reclassfile = getfile(window_title='Select point cloud to reclassify')

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
    default_values.model_inputs.append('veglab')

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
        print("\nModel name: {}".format(default_values.model_output_name))
        print("\nModel inputs:\n   {}".format(list(train_ins)))

    # build and train model
    mod,tt = build_model(model_name=str(default_values.model_output_name),
                        model_inputs=train_ins,
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

    # evaluate the model
    print('\nEvaluating model with validation set...')
    model_loss,model_accuracy = mod.evaluate(test_ds, verbose=2)

    # get today's date as string
    tdate = str(date.today()).replace('-','')

    # check if saved model dir already exists (create if not present)
    if not os.path.isdir('saved_models_'+tdate):
        os.makedirs('saved_models_'+tdate)

    print('\nWriting summary log file')
    # write model information and metadata to output txt file
    with open(os.path.join('saved_models_'+tdate,'SUMMARY_'+default_values.model_output_name+'.txt'),'w') as fh:
        # print model summary to output file
        # Pass the file handle in as a lambda function to make it callable
        mod.summary(print_fn=lambda x: fh.write(x+'\n'))
        fh.write('created: {}\n'.format(tdate))
        fh.write('bare-Earth file: {}\n'.format(default_values.filein_ground))
        fh.write('vegetation file: {}\n'.format(default_values.filein_vegetation))
        fh.write('model inputs: {}\n'.format(list(default_values.model_inputs)))
        fh.write('validation accuracy: {}\n'.format(model_accuracy))
        fh.write('validation loss: {}\n'.format(model_loss))
        fh.write('train time: {}'.format(datetime.timedelta(seconds=tt))+'\n')

    print('\nSaving model as .h5 and sub-directory in {}'.format('saved_models_'+tdate))
    # save the complete model (will create a new folder with the saved model)
    mod.save(os.path.join('saved_models_'+tdate,default_values.model_output_name))
    # save the model weights as H5 file
    mod.save(os.path.join('saved_models_'+tdate,(default_values.model_output_name+'.h5')))

    print('\nPlotting model')
    # plot the model as a PNG
    plot_model(mod, to_file=(os.path.join('saved_models_'+tdate,str(default_values.model_output_name)+'_GRAPH.png')), rankdir=default_values.plotdir, dpi=300)

    ## RECLASSIFY A FilE USING TRAINED MODEL FILE
    print('\n\nRECLASSIFYING FILE...')

    # load the model
    globals()[mod.name] = mod

    # get the model name from the loaded file
    default_values.model_output_name = mod.name

    ''' LOOK AT COMBINING MODEL_NAME WITH MODEL_FILE'''

    # get model inputs from the loaded file
    default_values.model_inputs = [f.name for f in mod.inputs]
    if verbose:
        print(default_values.model_inputs)

    # check if any geometry metrics are specified
    geomet = list(set(acceptablegeometrics).intersection(default_values.model_inputs))

    # print the model summary
    if verbose:
        print(mod.summary())

    # reclassify the input file
    try:
        predict_reclass_write(default_values.reclassfile,
                                [globals()[mod.name]],
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
    # create the arguments
    defs = Args('defs')

    main(default_values=defs)
