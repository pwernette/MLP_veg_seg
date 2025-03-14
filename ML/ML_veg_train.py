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


def ml_veg_train(default_values, verbose=True):
    # parse any command line arguments (if present)
    default_values.parse_cmd_arguments()

    if default_values.gui:
        request_window = App_train_only()
        request_window.create_widgets(default_values)
        request_window.mainloop()

    if verbose:
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

    print('\n\nMODEL INPUTS: {}'.format(list(default_values.model_inputs)))

    # the las2split() function performs the following actions:
    #   1) import training point cloud files
    #   2) compute vegetation indices
    #   3) split point clouds into training, testing, and validation sets
    if 'sd' in default_values.model_inputs or 'std' in default_values.model_inputs or 'stdev' in default_values.model_inputs:
        train_ds,val_ds,eval_ds,class_dat = las2split(default_values.filesin,
                                            veg_indices=default_values.model_vegetation_indices,
                                            class_imbalance_corr=default_values.training_class_imbalance_corr,
                                            training_split=default_values.training_split,
                                            data_reduction=default_values.training_data_reduction,
                                            geom_metrics='sd',
                                            xyz_mins=default_values.xyz_mins,
                                            xyz_maxs=default_values.xyz_maxs
                                            )
    else:
        train_ds,val_ds,eval_ds,class_dat = las2split(default_values.filesin,
                                                      veg_indices=default_values.model_vegetation_indices,
                                                      class_imbalance_corr=default_values.training_class_imbalance_corr,
                                                      training_split=default_values.training_split,
                                                      data_reduction=default_values.training_data_reduction,
                                                      xyz_mins=default_values.xyz_mins,
                                                      xyz_maxs=default_values.xyz_maxs
                                                      )
    print('\nClass dictionary:')
    [print(i,v) for i,v in enumerate(class_dat)]

    # get root directory
    default_values.rootdir = os.path.split(os.path.split(default_values.filesin[0])[0])[0]

    # append columns/variables of interest list
    # default_values.model_inputs.append('veglab')

    # print(train_ds)
    # print(train_ds.drop('veglab',axis=1).shape)

    # convert train, test, and validation to feature layers
    train_ds = df_to_dataset(train_ds, 
                             targetcolname='veglab',
                             label_depth=len(class_dat),
                             shuffle=default_values.training_shuffle,
                             cache_ds=default_values.training_cache,
                             prefetch=default_values.training_prefetch,
                             batch_size=default_values.training_batch_size)
    val_ds = df_to_dataset(val_ds, 
                           targetcolname='veglab',
                           label_depth=len(class_dat),
                           shuffle=default_values.training_shuffle,
                           cache_ds=default_values.training_cache,
                           prefetch=default_values.training_prefetch,
                           batch_size=default_values.training_batch_size)
    eval_ds = df_to_dataset(eval_ds, 
                            targetcolname='veglab',
                            label_depth=len(class_dat),
                            shuffle=default_values.training_shuffle,
                            cache_ds=default_values.training_cache,
                            prefetch=default_values.training_prefetch,
                            batch_size=default_values.training_batch_size)

    # get today's date as string
    tdate = str(date.today()).replace('-','')

    # check if saved model dir already exists (create if not present)
    if not os.path.isdir(os.path.join(default_values.rootdir,'saved_models_'+tdate)):
        os.makedirs(os.path.join(default_values.rootdir,'saved_models_'+tdate))

    # build and train model
    mod,history,tt = build_model(model_name=default_values.model_name,
                        training_tf_dataset=train_ds,
                        validation_tf_dataset=val_ds,
                        rootdirectory=os.path.join(default_values.rootdir,'saved_models_'+tdate),
                        nclasses=len(class_dat),
                        nodes=default_values.model_nodes,
                        activation_fx='relu',
                        dropout_rate=default_values.model_dropout,
                        loss_metric='mean_squared_error',
                        # loss_metric=tf.keras.losses.MeanSquaredError(),
                        model_optimizer='adam',
                        earlystopping=[default_values.model_early_stop_patience,default_values.model_early_stop_delta],
                        dotrain=True,
                        dotrain_epochs=default_values.training_epoch,
                        verbose=default_values.verbose_run)

    # evaluate the model
    print('\nEvaluating model with validation set...')
    model_eval = mod.evaluate(eval_ds, verbose=2)
    print('    loss: {}'.format(model_eval[0]))
    print('    cross entropy: {}'.format(model_eval[1]))
    print('    categorical accuracy: {}'.format(model_eval[2]))
    print('    precision: {}'.format(model_eval[3]))
    print('    recall: {}'.format(model_eval[4]))
    print('    AUC: {}'.format(model_eval[5]))

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
        f1.plot(history.history['cat_accuracy'])
        f1.plot(history.history['val_cat_accuracy'])
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
        fh.write('input point cloud files: {}\n'.format(default_values.filesin))
        
        fh.write('\nvegetation indices: {}\n'.format(list(default_values.model_vegetation_indices)))
        fh.write('model inputs: {}\n'.format(list(default_values.model_inputs)))

        fh.write('\vEvaluation metrics:\n')
        fh.write('Loss: {}\n'.format(model_eval[0]))
        fh.write('Cross Entropy: {}\n'.format(model_eval[1]))
        fh.write('Categorical Accuracy: {}\n'.format(model_eval[2]))
        fh.write('Precision: {}\n'.format(model_eval[3]))
        fh.write('Recall: {}\n'.format(model_eval[4]))
        fh.write('AUC: {}\n'.format(model_eval[5]))

        fh.write('\ntrain time: {}'.format(datetime.timedelta(seconds=tt))+'\n')
        fh.write('\nClass Dictionary:\n')
        for key, value in class_dat.items():  
            fh.write('%i: %s\n' % (value, key))

    print('\nSaving model as .h5 and sub-directory in {}...'.format(os.path.join(default_values.rootdir,'saved_models_'+tdate)))
    # save the complete model (will create a new folder with the saved model)
    #mod.save(os.path.join(default_values.rootdir,'saved_models_'+tdate,default_values.model_name))
    # save the model and model weights as H5 files
    if float((tf.__version__).split('.',1)[1]) >= 11.0:
        mod.save(os.path.join(default_values.rootdir,'saved_models_'+tdate,(default_values.model_name+'_FULL_MODEL.keras')))
    else:
        mod.save(os.path.join(default_values.rootdir,'saved_models_'+tdate,(default_values.model_name+'_FULL_MODEL.h5')))
    mod.save_weights(os.path.join(default_values.rootdir,'saved_models_'+tdate,(default_values.model_name+'_MODEL_WEIGHTS.h5')))

if __name__ == '__main__':
    defs = Args_train_only('defs')

    # try:
    ml_veg_train(default_values=defs)
    # except:
    #     traceback.print_exc()
