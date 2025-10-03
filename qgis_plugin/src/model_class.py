import time, sys, os, datetime
from datetime import datetime

import numpy as np
import pandas as pd

# load plotting module
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import clear_output

# scikit learn is used to split the pandas.DataFrame to train, test, and val
from sklearn.model_selection import train_test_split

# load Tensorflow modules
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

# import laspy and check major version
import laspy
if int(laspy.__version__.split('.')[0]) == 1:
    # in major version laspy 1.x.x files are read using laspy.file.File
    from laspy import file as lf
else:
    import lazrs

from IPython.display import clear_output

from .argclass import *
from .vegindex import *


class Classifier():
    def __init__(self,default_arguments_object):
        self.das = default_arguments_object
        # print TF information to screen
        if self.das.model_verbose_run >= 0:
            print('Tensorflow Information:')
            print("   TF Version: ", tf.__version__)
            print("   Eager mode: ", tf.executing_eagerly())
            print("   Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            print("   GPU name: ", tf.config.experimental.list_physical_devices('GPU'))
            print("laspy Information:")
            # print laspy version installed and configured
            print("   laspy Version: {}\n".format(laspy.__version__))

        # ensure that vegetation indices/inputs are present
        if self.das.model_vegetation_indices == '' or self.das.model_vegetation_indices == 'NA':
            print('No vegetation index/indices specified, defaulting to RGB only.')
            self.das.model_vegetation_indices = 'rgb'

        #here

    def read_laz(self):
        '''
        Read two LAS/LAZ point clouds representing a sample of ground and vegetation
        points. Because laspy v1.x and v2.x read compressed LAZ files differently, 
        the approach to reading in data varies.
        '''
        if hasattr(self.das,'filein_ground') and hasattr(self.das,'filein_ground'):
            if self.das.laspy_version == 1:
                try:
                    self.pc.ground = lf.File(self.das.filein_ground,mode='r')
                    self.pc.veg = lf.File(self.das.filein_vegetation,mode='r')
                    print('Read {}'.format(self.das.filein_ground))
                    print('Read {}'.format(self.das.filein_vegetation))
                except Exception as e:
                    print('ERROR: Unable to read point cloud file(s). See error below for more information.')
                    sys.exit(e)
            elif self.das.laspy_version == 2:
                try:
                    self.pc.ground = laspy.read(self.das.filein_ground)
                    self.pc.veg = laspy.read(self.das.filein_vegetation)
                    print('Read {}'.format(self.das.filein_ground))
                    print('Read {}'.format(self.das.filein_vegetation))
                except Exception as e:
                    print('ERROR: Unable to read point cloud file(s). See error below for more information.')
                    sys.exit(e)
        else:
            sys.exit('\nERROR: One or more training point clouds are not specified.')

    def split_training_point_clouds(self):
        '''
        If vegetation indices are specified by veg_indices, then the defined
        indices are computed for each input point cloud. The data is, by default,
        checked for class imbalance based on the size of the two point clouds, and,
        if there is an imbalance, the larger point cloud is shuffled and randomly
        sampled to the same size as the smaller point cloud. There is also an option
        to reduce the data volume to a user-specified proportion of the class
        imbalance corrected point clouds. Finally, both point clouds are randomly
        split into a training, testing, and validation pandas.DataFrame object and
        a vegetation label field is placed on each class accordingly.
        '''
        if hasattr(self.pc,'ground') and hasattr(self.pc,'veg') and hasattr(self.das,'model_vegetation_indices') and hasattr(self.das,'training_split'):
            # compute vegetation indices using vegidx() function (from vegindex.py)
            self.ground.names,self.ground.dat = vegidx(self.pc.ground, indices=self.das.model_vegetation_indices, geom_metrics=self.das.geometry_metrics)
            self.veg.names,self.veg.dat = vegidx(self.pc.veg, indices=self.das.model_vegetation_indices, geom_metrics=self.das.geometry_metrics)

            # transpose the output objects
            self.ground.dat = np.transpose(self.ground.dat)
            self.veg.dat = np.transpose(self.veg.dat)

            # clean up workspace/memory
            if self.das.laspy_version == 1:
                try:
                    self.pc.ground.close()
                    self.pc.veg.close()
                except Exception as e:
                    print(e)
                    pass

            # OPTIONAL: print number of points in each input dense point cloud
            if self.das.verbose_run > 0:
                print('# ground points     = {}'.format(self.ground.dat.shape))
                print('# vegetation points = {}'.format(self.veg.dat.shape))

            # sample larger dat to match size of smaller dat
            if self.das.training_class_imbalance_corr:
                if self.ground.dat.shape[0]>self.veg.dat.shape[0]:
                    self.ground.dat = train_test_split(self.ground.dat, train_size=self.veg.dat.shape[0]/self.ground.dat.shape[0], random_state=42)[0]
                elif self.veg.dat.shape[0]>self.ground.dat.shape[0]:
                    self.veg.dat = train_test_split(self.veg.dat, train_size=self.ground.dat.shape[0]/self.veg.dat.shape[0], random_state=42)[0]

            # sub-sample the vegetation and no-vegetation data to cut the data volume
            if self.das.training_data_reduction<1.0:
                # data reduction to the user-specified proportion
                self.ground.dat = train_test_split(self.ground.dat, train_size=self.das.training_data_reduction, random_state=123)[0]  # sub-sample no-veg points
                self.veg.dat = train_test_split(self.veg.dat, train_size=self.das.training_data_reduction, random_state=123)[0]  # sub-sample veg points

            # convert each of the samples to pandas.DataFrame objects for subsampling
            self.ground.pd = pd.DataFrame(self.ground.dat.astype('float16'), columns=self.ground.names)
            self.veg.pd = pd.DataFrame(self.veg.dat.astype('float16'), columns=self.veg.names)

            # append vegetation label column to pd.DataFrame
            self.gound.pd['veglab'] = np.full(shape=self.veg.dat.shape[0], fill_value=0, dtype=np.float16)
            self.veg.pd['veglab'] = np.full(shape=self.ground.dat.shape[0], fill_value=1, dtype=np.float16)

            # add a "veglab" column to represent vegetation labels
            self.ground.names = np.append(self.ground.names, 'veglab')
            self.veg.names = np.append(self.veg.names, 'veglab')

            # training, testing, and validation splitting
            train_g,val_g,train_v,val_v = train_test_split(self.gound.pd, self.veg.pd, train_size=self.das.training_split, shuffle=True, random_state=42)
            train_g,test_g,train_v,test_v = train_test_split(val_g, val_v, train_size=0.5, shuffle=True, random_state=42,)

            # concatenate ground and veg pd.DataFrame objects
            self.dat.train = pd.concat([train_g,train_v], ignore_index=True)
            self.dat.test = pd.concat([test_g,test_v], ignore_index=True)
            self.dat.val = pd.concat([val_g,val_v], ignore_index=True)

            # shuffle dataframes
            if self.das.training_shuffle:
                self.dat.train = self.dat.train.sample(frac=1).reset_index(drop=True)
                self.dat.test = self.dat.test.sample(frac=1).reset_index(drop=True)
                self.dat.val = self.dat.val.sample(frac=1).reset_index(drop=True)

            # clean up memory/workspace
            del(train_g,train_v,test_g,test_v,val_g,val_v)

            # OPTIONAL: print info about training, testing, validation split numbers
            if self.das.verbose_run > 0:
                print('  {} train examples'.format(len(self.dat.train)))
                print('  {} validation examples'.format(len(self.dat.val)))
                print('  {} test examples'.format(len(self.dat.test)))
        else:
            sys.exit('\nERROR: Unable to split training clouds.')

    def convert_training_data_to_tensors(self):
        '''
        Read a pandas.DataFrame object and (1) convert to tf.data object, (2) return
        a list of column names from the pd.DataFrame, and (3) return a
        tf.DenseFeatures layer.
        '''
        
        if hasattr(self.dat,'train') and hasattr(self.dat,'val') and hasattr(self.dat,'test') and hasattr(self.das,'model_inputs'):
            """ Convert the pd.DataFrame to tf.Dataset objects """
            """ ACTIVELY SEEKING MORE ELOQUENT APPROACH TO THE FOLLOWING: """
            if 'veglab' in self.dat.train.columns:
                # extract the labels from the input data
                labs = ds.pop('veglab')
                # convert the pd.DataFrame objects to tf.Tensor objects
                self.dat.train = tf.data.Dataset.from_tensors(tf.convert_to_tensor(self.dat.train))
                labs = tf.one_hot(labs, depth=self.das.n_classes)
                # package the input data and labels in a single tf.Tensor object
                self.dat.train = tf.data.Dataset.from_tensors((self.dat.train,labs))
            else:
                self.dat.train = tf.data.Dataset.from_tensors(tf.convert_to_tensor(self.dat.train))

            if 'veglab' in self.dat.val.columns:
                # extract the labels from the input data
                labs = ds.pop('veglab')
                # convert the pd.DataFrame objects to tf.Tensor objects
                self.dat.val = tf.data.Dataset.from_tensors(tf.convert_to_tensor(self.dat.val))
                labs = tf.one_hot(labs, depth=self.das.n_classes)
                # package the input data and labels in a single tf.Tensor object
                self.dat.val = tf.data.Dataset.from_tensors((self.dat.val,labs))
            else:
                self.dat.val = tf.data.Dataset.from_tensors(tf.convert_to_tensor(self.dat.val))

            if 'veglab' in self.dat.test.columns:
                # extract the labels from the input data
                labs = ds.pop('veglab')
                # convert the pd.DataFrame objects to tf.Tensor objects
                self.dat.test = tf.data.Dataset.from_tensors(tf.convert_to_tensor(self.dat.test))
                labs = tf.one_hot(labs, depth=self.das.n_classes)
                # package the input data and labels in a single tf.Tensor object
                self.dat.test = tf.data.Dataset.from_tensors((self.dat.test,labs))
            else:
                self.dat.test = tf.data.Dataset.from_tensors(tf.convert_to_tensor(self.dat.test))


            print('\nPre-processing pd.DataFrame objects to tf.Tensor objects.')

            if self.das.verbose_run > 0:
                print('  Column names: {}'.format(self.das.model_inputs))

            # shuffle the training data
            if self.das.training_shuffle:
                print('  Shuffling training dataset...')
                self.dat.train = self.dat.train.shuffle(buffer_size=tf.data.experimental.cardinality(self.dat.train).numpy()+1)
                self.dat.val = self.dat.train.shuffle(buffer_size=tf.data.experimental.cardinality(self.dat.val).numpy()+1)
                self.dat.test = self.dat.train.shuffle(buffer_size=tf.data.experimental.cardinality(self.dat.test).numpy()+1)
        
            # batch, cache, and prefetch the Tensor (as specified by inputs)
            if self.das.training_batch_size > 1:
                print('  Batching dataset to batch size {}...'.format(self.das.training_batch_size))
                self.dat.train = self.dat.train.batch(self.das.training_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
                self.dat.val = self.dat.val.batch(self.das.training_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                self.dat.test = self.dat.test.batch(self.das.training_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            if self.das.training_cache:
                print('  Caching dataset...')
                self.dat.train = self.dat.train.cache()
                self.dat.val = self.dat.val.cache()
                self.dat.test = self.dat.test.cache()
            if self.das.training_prefetch:
                print('  Prefetching dataset using AUTOTUNE...')
                self.dat.train = self.dat.train.prefetch(tf.data.experimental.AUTOTUNE)
                self.dat.val = self.dat.val.prefetch(tf.data.experimental.AUTOTUNE)
                self.dat.test = self.dat.test.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            sys.exit('\nERROR: Unable to convert training data to tensors')

    def build_model(self):
        #model_inputs, input_feature_layer, training_tf_dataset,  # replace input_tf_dataset with this line
        if hasattr(self.dat,'train') and hasattr(self.das,'model_nodes') and hasattr(self.das,'model_activation_function'):
            # get the input data shape
            try:
                dshape = self.dat.train.element_spec.shape
                print('Input data shape: {}'.format(dshape))
            except:
                dshape = self.dat.train.element_spec[0].shape
                print('Input data shape: {}'.format(dshape))

            print('Building {} model...\n'.format(self.das.model_output_name))
            # the first layer should take the input features as its input
            inputs = {
                'r': Input(shape=(1,), name='r'),
                'g': Input(shape=(1,), name='g'),
                'b': Input(shape=(1,), name='b')
                }
            input_layer = Input(shape=dshape, name='input_points')
            # l = layers.Dense(nodes[0], activation=activation_fx)(input_feat_layer)
            if len(self.das.model_nodes)<1:
                print('WARNING: No nodes specified, defaulting to 3 layers with 8 nodes each.')
                self.das.model_nodes = [8,8,8]
            l = Dense(self.das.model_nodes[0], activation=self.das.model_activation_function, name=('l0_'+str(self.das.model_nodes[0])+'_nodes'))(input_layer)
            # each subsequent layer (if present) should take the preceeding layer as its input
            if len(self.das.model_nodes)>1:
                for c,n in enumerate(self.das.model_nodes[1:]):
                    l = Dense(n, activation=self.das.model_activation_function, name=('l'+str(c+1)+'_'+str(n)+'_nodes'))(l)
            # add a dropout layer to reduce the chance of overfitting
            l = Dropout(self.das.model_dropout, name='dropout')(l)
            # flatten the output to a single Dense layer
            flatten_layer = Flatten(name='flattened_dat')(l)
            out_classes = Dense(2, name='multiple_class_labels')(flatten_layer)
            out = Activation('softmax', dtype='float32', name='softmax_class_label')(out_classes)

            # build the model
            self.mod = tf.keras.Model(inputs=input_layer, outputs=out, name=self.das.model_output_name)
        else:
            sys.exit('\nERROR: Unable to build model')
       
    def compile_model(self):
        # dictionary of tf.keras training loss functions
        # NOTE: future versions will not support 'mean' or 'mean_squared_error'
        loss_fx_dict = {
            'categorical': tf.keras.losses.CategoricalCrossentropy(),
            'binary': tf.keras.losses.BinaryCrossentropy(),
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
            'sparse': tf.keras.losses.SparseCategoricalCrossentropy(),
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'mean': tf.keras.losses.MeanSquaredError()
        }
        
        if hasattr(self,'mod') and hasattr(self.das,'model_metric') and hasattr(self.das,'model_optimizer'):
            # compile the model
            if 'categorical' in self.das.model_metric:
                mod_met = 'categorical_crossentropy'
            else:
                mod_met = 'accuracy'
            try:
                self.mod.compile(loss=loss_fx_dict[self.das.model_metric],
                              optimizer=self.das.model_optimizer,
                              metrics=[mod_met],
                              run_eagerly=True)
            except Exception as e:
                print('FATAL ERROR: Failed to compile TF model.')
                sys.exit(e)

            # optional: print the model summary (includes structure)
            if self.das.verbose_run > 0:
                print(self.mod.summary())
        else:
            sys.exit('\nERROR: Unable to compile model.')
            
    def save_model_summary(self):
        """ If the output_models directory does not exist, then create it """
        if not os.path.isdir(os.path.join(self.das.rootdir,'output_models')):
            os.mkdir(os.path.join(self.das.rootdir, 'output_models'))
        """ Write a text file with the trained model summary """
        try:
            with open(os.path.join(self.das.rootdir,'output_models',self.das.model_name+'.txt'), 'w') as summaryout:
                self.mod.summary(print_fn=lambda x: summaryout.write(x+'\n'))
        except Exception as e:
            sys.exit(e)

    def train_model(self):
        if hasattr(self,'mod'):
            # create history callback
            call_list = []

            # add early stopping to the callback list
            call_list.append(EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_early_stop_patience,
                    min_delta=self.model_early_stop_delta,
                    mode='max'))

            # add learning rate scheduler to callback list
            call_list.append(ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5, 
                    patience=7, 
                    min_delta=1e-4, 
                    mode='max', 
                    verbose=1))

            # expand the tf.Tensor dimensions
            #training_dat = training_dat.map(lambda x: (tf.expand_dims(x, axis=0)))
            #validation_dat = validation_dat.map(lambda x: (tf.expand_dims(x, axis=0)))
            #training_dat = training_dat.batch(1000)
            #validation_dat = validation_dat.batch(1000)
        
            start_time = datetime.now()
            print('Training {} model. Starting at {}'.format(self.mod.name, start_time.strftime('\n%Y/%m/%d %H:%M:%S\n')))
            start_time = time.time()
        
            # train the model using training_dat and validate against validation_dat
            self.history = self.mod.fit(self.dat.train,
                                    validation_data=self.dat.val,
                                    batch_size=self.das.training_batch_size,
                                    callbacks=[call_list],
                                    epochs=self.das.training_epoch,
                                    verbose=self.das.verbose_run,
                                    use_multiprocessing=True
            )
            train_time = time.time()-start_time
            print("Train time = {}s".format(train_time))

            print('\nPreliminary model results:')
            print(self.mod.get_metrics_result())

    def evaluate_model(self):
        if hasattr(self,'mod') and hasattr(self.dat,'test'):
            """ Evaluate the model """
            self.model_eval = self.mod.evaluate(self.dat.test, batch_size=32, verbose=2)
            print('  Validation Loss, Validation Accuracy: {}'.format(self.model_eval))

    def save_model(self):
        print(1)

    def calculate_confusion_matrix(self):
        if hasattr(self,'mod') and hasattr(self.dat,'test') and hasattr(self.das,'model_verbose_run'):
            """ Iterate through the training data """
            for x, y in self.dat.test:
                """ Extract the training labels """
                original_labels = np.concatenate([original_labels, y])
                """ Predict class using model """
                model_predictions = np.concatenate([model_predictions, tf.argmax(self.mod.predict(x, verbose=self.das.model_verbose_run), axis=-1)])

            """ Create the confusion matrix (and calculate as a percentage) """
            confusion_mat = tf.math.confusion_matrix(labels=original_labels, predictions=model_predictions)

            """ WRITE CONFUSION MATRIX OUT """