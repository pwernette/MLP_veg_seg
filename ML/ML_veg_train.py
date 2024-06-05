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
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *

# import functions from other files
from src.fileio import *
from src.tk_get_user_input_TRAIN_ONLY import *
from src.vegindex import *
from src.miscfx import *
from src.modelbuilder import *


def main(default_values, verbose=True):
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

    # the las2split() function performs the following actions:
    #   1) import training point cloud files
    #   2) compute vegetation indices
    #   3) split point clouds into training, testing, and validation sets
    if 'sd' in default_values.model_inputs:
        train_ds,test_ds,val_ds = las2split(default_values.filein_ground,
                               default_values.filein_vegetation,
                               veg_indices=default_values.model_vegetation_indices,
                               class_imbalance_corr=default_values.training_class_imbalance_corr,
                               training_split=default_values.training_split,
                               data_reduction=default_values.training_data_reduction,
                               geom_metrics='sd')
    else:
        train_ds,test_ds,val_ds = las2split(default_values.filein_ground,
                               default_values.filein_vegetation,
                               veg_indices=default_values.model_vegetation_indices,
                               class_imbalance_corr=default_values.training_class_imbalance_corr,
                               training_split=default_values.training_split,
                               data_reduction=default_values.training_data_reduction)

    # get root directory
    default_values.rootdir = os.path.split(os.path.split(default_values.filein_ground)[0])[0]

    # append columns/variables of interest list
    default_values.model_inputs.append('veglab')

    # convert train, test, and validation to feature layers
    train_ds = df_to_dataset(train_ds, 'veglab',
                             shuffle=default_values.training_shuffle,
                             cache_ds=default_values.training_cache,
                             prefetch=default_values.training_prefetch,
                             batch_size=default_values.training_batch_size)
    val_ds = df_to_dataset(val_ds, 'veglab',
                             shuffle=default_values.training_shuffle,
                             cache_ds=default_values.training_cache,
                             prefetch=default_values.training_prefetch,
                             batch_size=default_values.training_batch_size)
    test_ds = df_to_dataset(test_ds, 'veglab',
                             shuffle=default_values.training_shuffle,
                             cache_ds=default_values.training_cache,
                             prefetch=default_values.training_prefetch,
                             batch_size=default_values.training_batch_size)
    #train_ds,train_ins,train_lyr = pd2fl(train, default_values.model_inputs,
    #                                        shuf=default_values.training_shuffle,
    #                                        ds_prefetch=default_values.training_prefetch,
    #                                        batch_sz=default_values.training_batch_size,
    #                                        ds_cache=default_values.training_cache)
    #val_ds,_,_ = pd2fl(val, default_values.model_inputs,
    #                                        shuf=default_values.training_shuffle,
    #                                        ds_prefetch=default_values.training_prefetch,
    #                                        batch_sz=default_values.training_batch_size,
    #                                        ds_cache=default_values.training_cache)
    #test_ds,_,_ = pd2fl(test, default_values.model_inputs,
    #                                        shuf=default_values.training_shuffle,
    #                                        ds_prefetch=default_values.training_prefetch,
    #                                        batch_sz=default_values.training_batch_size,
    #                                        ds_cache=default_values.training_cache)

    # print model input attributes
    #if verbose:
    #    print("\nModel inputs:\n   {}".format(list(train_ins)))

    # build and train model
    mod,history,tt = build_model(model_name=default_values.model_name,
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
                        verbose=default_values.verbose_run)

    # evaluate the model
    print('\nEvaluating model with validation set...')
    model_eval = mod.evaluate(test_ds, verbose=2)

    # get today's date as string
    tdate = str(date.today()).replace('-','')

    # check if saved model dir already exists (create if not present)
    if not os.path.isdir(os.path.join(default_values.rootdir,'saved_models_'+tdate)):
        os.makedirs(os.path.join(default_values.rootdir,'saved_models_'+tdate))

    if default_values.training_plot:
        print('\nPlotting model...')
        # plot the model as a PNG
        plot_model(mod, to_file=(os.path.join(os.path.join(default_values.rootdir,'saved_models_'+tdate),str(default_values.model_name)+'_MODEL_ARCHITECTURE.png')), rankdir=default_values.plotdir, dpi=300)

        # set default matplotlib parameters similar to IPython
        mpl.rcParams['figure.figsize'] = (12.0,6.0)
        mpl.rcParams['font.size'] = 11
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['figure.subplot.bottom'] = 0.125

        # create the figure
        fig,(f1,f2) = plt.subplots(1,2)
        fig.suptitle(default_values.model_name+' Training History')

        # plot accuracy
        f1.plot(history.history['accuracy'])
        f1.plot(history.history['val_accuracy'])
        f1.set_title('Model Accuracy')
        f1.set(xlabel='epoch', ylabel='accuracy')
        f1.legend(['train','test'], loc='right')

        # plot loss
        f2.plot(history.history['loss'])
        f2.plot(history.history['val_loss'])
        f2.set_title('Model Loss')
        f2.set(xlabel='epoch', ylabel='loss')
        f2.legend(['train','test'], loc='right')

        # save the figure
        fig.savefig(os.path.join(default_values.rootdir,'saved_models_'+tdate,default_values.model_name+'_PLOT_TRAINING.png'))

    print('\nWriting summary log file...')
    # write model information and metadata to output txt file
    with open(os.path.join(default_values.rootdir,'saved_models_'+tdate,default_values.model_name+'_MODEL_SUMMARY.txt'),'w') as fh:
        # print model summary to output file
        # Pass the file handle in as a lambda function to make it callable
        mod.summary(print_fn=lambda x: fh.write(x+'\n'))
        fh.write('created: {}\n'.format(tdate))
        fh.write('bare-Earth file: {}\n'.format(default_values.filein_ground))
        fh.write('vegetation file: {}\n'.format(default_values.filein_vegetation))
        fh.write('model inputs: {}\n'.format(list(default_values.model_inputs)))
        fh.write('validation accuracy: {}\n'.format(model_eval[1]))
        fh.write('validation loss: {}\n'.format(model_eval[0]))
        fh.write('train time: {}'.format(datetime.timedelta(seconds=tt))+'\n')

    print('\nSaving model as .h5 and sub-directory in {}...'.format(os.path.join(default_values.rootdir,'saved_models_'+tdate)))
    # save the complete model (will create a new folder with the saved model)
    #mod.save(os.path.join(default_values.rootdir,'saved_models_'+tdate,default_values.model_name))
    # save the model and model weights as H5 files
    mod.save(os.path.join(default_values.rootdir,'saved_models_'+tdate,(default_values.model_name+'_FULL_MODEL.h5')))
    mod.save_weights(os.path.join(default_values.rootdir,'saved_models_'+tdate,(default_values.model_name+'_MODEL_WEIGHTS.h5')))

if __name__ == '__main__':
    defs = Args('defs')
    defs.parse_cmd_arguments()

    if defs.gui:
        foo = App()
        foo.create_widgets(defs)
        foo.mainloop()

    main(default_values=defs)
