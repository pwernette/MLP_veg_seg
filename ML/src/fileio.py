import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import date

# import laspy and check major version
import laspy
if int(laspy.__version__.split('.')[0]) == 1:
    # in major version laspy 1.x.x files are read using laspy.file.File
    from laspy import file

# scikit learn is used to split the pandas.DataFrame to train, test, and val
from sklearn.model_selection import train_test_split

# load Tensorflow modules
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

# load tkinter modules for GUI user input
import tkinter as tk
from tkinter import *

# load custom modules
from .vegindex import vegidx
from .miscfx import *
from .modelbuilder import *


def getfile(window_title='Select File'):
    '''
    Function to open a dialog window where the user can select a single file.
    '''
    root_win = Tk()  # initialize the tKinter window to get user input
    root_win.withdraw()
    root_win.update()
    file_io = askopenfile(title=window_title)  # get user input for the output directory
    root_win.destroy()  # destroy the tKinter window
    if not os.name=='nt':  # if the OS is not Windows, then correct slashes in directory name
        file.io = ntpath.normpath(file_io)
    return str(file_io.name)

def getmodelname():
    '''
    Function to create a dialog box to get a user-specified name for the model
    to be created.

    Returns the model name as a string (with underscores in place of spaces)
    '''
    # create window instance
    win = Tk()
    win.title('Specify Model Name')

    # create entry widget to accept user input
    textbox = Text(win, height=1, width=50)
    textbox.pack()

    def getinput():
        global modelname
        modelname = textbox.get('1.0','end-1c').split('\n')[0]
        win.destroy()
        return modelname

    def cancel_and_exit():
        win.destroy()
        sys.exit('Exiting program.')

    # create validation button
    buttonconfirm = Button(win,
                            text='Confirm Model Name',
                            width=40,
                            command=lambda:getinput())
    buttonconfirm.pack(pady=5)

    win.bind('<Return>', lambda event:getinput())

    win.bind('<Escape>', lambda event:cancel_and_exit())

    win.mainloop()
    if ' ' in modelname:
        mname = modelname.replace(' ','_')
    else:
        mname = modelname
    return(mname)

def las2split(infile_ground_pc, infile_veg_pc,
                veg_indices=[], geometry_metrics=[],
                training_split=0.7, class_imbalance_corr=True,
                data_reduction=1.0, verbose=True):
    '''
    Read two LAS/LAZ point clouds representing a sample of ground and vegetation
    points. If vegetation indices are specified by veg_indices, then the defined
    indices are computed for each input point cloud. The data is, by default,
    checked for class imbalance based on the size of the two point clouds, and,
    if there is an imbalance, the larger point cloud is shuffled and randomly
    sampled to the same size as the smaller point cloud. There is also an option
    to reduce the data volume to a user-specified proportion of the class
    imbalance corrected point clouds. Finally, both point clouds are randomly
    split into a training, testing, and validation pandas.DataFrame object and
    a vegetation label field is placed on each class accordingly.

    Inputs:

    :param pandas.DataFrame input_pd_dat: Input pandas DataFrame
    :param list col_names: Column names of interest from input_pd_dat
    :param string targetcol: Target column to use for training
    :param float training_split: Proportion of data to use for training
        (Remainder of the data will be used for validation)
    :param bool class_imbalance_corr: Option to correct for class imbalance
    :param float data_reduction: Proportion to reduce the data volume to
    :param bool verbose: Option to print information to the console

    Returns:

        * trainout (:py:class:`pd.DataFrame`)
        * testout (:py:class:`pd.DataFrame`)
        * valout (:py:class:`pd.DataFrame`)
    '''
    laspy_majorversion = int(laspy.__version__.split('.')[0])
    # open both ground and vegetation files
    if laspy_majorversion == 1:
        try:
            fground = file.File(infile_ground_pc,mode='r')
            fveg = file.File(infile_veg_pc,mode='r')
        except Exception as e:
            sys.exit(e)
    elif laspy_majorversion == 2:
        try:
            fground  =laspy.read(infile_ground_pc)
            fveg = laspy.read(infile_veg_pc)
        except Exception as e:
            sys.exit(e)

    # compute vegetation indices
    print('Read {}'.format(infile_ground_pc))
    names_ground,dat_ground = vegidx(fground, indices=veg_indices, geom_metrics=geometry_metrics)
    print('Read {}'.format(infile_veg_pc))
    names_veg,dat_veg = vegidx(fveg, indices=veg_indices, geom_metrics=geometry_metrics)

    # transpose the output objects
    ground_sample = np.transpose(dat_ground)
    veg_sample = np.transpose(dat_veg)

    # clean up memory
    del(dat_ground,dat_veg)

    # add a "veglab" column to represent vegetation labels
    names_ground = np.append(names_ground, 'veglab')
    names_veg = np.append(names_veg, 'veglab')

    # clean up workspace/memory
    if laspy_majorversion == 1:
        try:
            fground.close()
            fveg.close()
        except Exception as e:
            print(e)
            pass

    # OPTIONAL: print number of points in each input dense point cloud
    if verbose:
        print('# of ground points     = {}'.format(ground_sample.shape))
        print('# of vegetation points = {}'.format(veg_sample.shape))

    # sample larger dat to match size of smaller dat
    if class_imbalance_corr:
        if ground_sample.shape[0]>veg_sample.shape[0]:
            ground_sample = train_test_split(ground_sample, train_size=veg_sample.shape[0]/ground_sample.shape[0], random_state=42)[0]
        elif veg_sample.shape[0]>ground_sample.shape[0]:
            veg_sample = train_test_split(veg_sample, train_size=ground_sample.shape[0]/veg_sample.shape[0], random_state=42)[0]

    # sub-sample the vegetation and no-vegetation data to cut the data volume
    if data_reduction<1.0:
        # data reduction to the user-specified proportion
        ground_sample = train_test_split(ground_sample, train_size=data_reduction_percent, random_state=42)[0]  # sub-sample no-veg points
        veg_sample = train_test_split(veg_sample, train_size=data_reduction_percent, random_state=42)[0]  # sub-sample veg points

    # convert each of the samples to pandas.DataFrame objects for subsampling
    pd_ground = pd.DataFrame(ground_sample.astype('float32'), columns=names_veg[:-1])
    pd_veg = pd.DataFrame(veg_sample.astype('float32'), columns=names_veg[:-1])

    # append vegetation label column to pd.DataFrame
    pd_ground['veglab'] = np.full(shape=veg_sample.shape[0], fill_value=0, dtype=np.float32)
    pd_veg['veglab'] = np.full(shape=ground_sample.shape[0], fill_value=1, dtype=np.float32)

    # clean up memory/workspace
    del(ground_sample, veg_sample)

    # training, testing, and validation splitting
    train_g,test_g,train_v,test_v = train_test_split(pd_ground, pd_veg, train_size=training_split, random_state=42)
    train_g,val_g,train_v,val_v = train_test_split(train_g, train_v, train_size=training_split, random_state=42)

    # concatenate ground and veg pd.DataFrame objects
    trainout = pd.concat([train_g,train_v], ignore_index=True)
    testout = pd.concat([test_g,test_v], ignore_index=True)
    valout = pd.concat([val_g,val_v], ignore_index=True)

    # clean up memory/workspace
    del(train_g,train_v,test_g,test_v,val_g,val_v)

    # OPTIONAL: print info about training, testing, validation split numbers
    if verbose:
        print('  {} train examples'.format(len(trainout)))
        print('  {} validation examples'.format(len(valout)))
        print('  {} test examples'.format(len(testout)))

    # return the train, test, and validationn objects
    # all outputs are pandas.DataFrame objects
    return trainout,testout,valout


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, targetcolname='', shuffle=False, prefetch=False, cache_ds=False, batch_size=32):
    '''
    Read a pandas.DataFrame object and convert to tf.data object.

    Depending on the arguments specified, (1) the pandas.DataFrame may be
    shuffled, (2) prefetching may occur to improve training time, and (3) data
    may be cached in memory temporarily. Regardless of the arguments specified,
    the data will be broken up into the specified batch size.

    Inputs:

    :param pandas.DataFrame dataframe: Input pandas DataFrame
    :param string targetcolname: Column names of interest from input_pd_dat
    :param bool shuffle: Optional argument to shuffle input pd.DataFrame
    :param bool prefetch: Optional argument to prefetch batches
        (This MAY speed up training where fetching takes a long time)
    :param bool cache_ds: Optional argument to cache batches
        (In cases where fetching takes a long time, this may speed up training)
        however, it will likely use more temporary memory for caching)
    :param int batch_size: Optional argument to specify batch size

    Returns:

        * ds (:py:class:`tf.Dataset`)
    '''
    dataframe = dataframe.copy()
    if shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    if not targetcolname == '':
        labels = dataframe.pop(targetcolname)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    if cache_ds:
        ds = ds.cache()
    return ds


def pd2fl(input_pd_dat,
            col_names=['r','g','b'],
            targetcol='',
            dat_type='float32',
            shuf=False,
            batch_sz=32,
            ds_prefetch=False,
            ds_cache=False,
            verbose=False):
    '''
    Read a pandas.DataFrame object and (1) convert to tf.data object, (2) return
    a list of column names from the pd.DataFrame, and (3) return a
    tf.DenseFeatures layer.

    Inputs:

    :param pandas.DataFrame input_pd_dat: Input pandas DataFrame
    :param list col_names: Column names of interest from input_pd_dat
    :param string targetcol: Target column to use for training
    :param string dat_type: Data format to coerce input data to
    :param bool shuf: Optional argument to shuffle input pd.DataFrame
    :param int batch_sz: Optional argument to specify batch size
    :param bool ds_prefetch: Optional argument to prefetch batches
        (This MAY speed up training where fetching takes a long time)
    :param bool ds_cache: Optional argument to cache batches
        (In cases where fetching takes a long time, this may speed up training)
        however, it will likely use more temporary memory for caching)
    :param bool verbose: Optional argument to print information to console

    Returns:

        * dset (:py:class:`tf.Dataset`)
        * inpts (:py:class:`list`)
        * lyr (:py:class:`tf.DenseFeatures`)
    '''
    # special use cases:
    #  CASE 1: use all vegetation indices
    if col_names == 'all':
        col_names.remove('all')
        col_names = list(input_pd_dat.columns)
    #  CASE 2: use RGB and "simple" vegetation indices
    if 'simple' in col_names:
        col_names.remove('simple')
        col_names = ['r','g','b','exr','exg','exb','exgr'] + col_names

    if targetcol == '' and 'veglab' in col_names:
        targetcol = 'veglab'

    # print(col_names)
    dset = df_to_dataset(input_pd_dat[col_names].astype(dat_type),
                                 targetcolname=targetcol,
                                 shuffle=shuf,
                                 prefetch=ds_prefetch,
                                 batch_size=batch_sz,
                                 cache_ds=ds_cache)
    # need to drop 'veglab' from the column names, if present
    if 'veglab' in col_names:
        col_names.remove('veglab')

    # create a list of feature columns
    feat_cols = []
    for header in col_names:
        feat_cols.append(feature_column.numeric_column(header))
    # option to print feature columns to console
    if verbose:
        for i in feat_cols:
            print(i)

    # create tf.DenseFeatures layer from feature columns
    feat_lyr = tf.keras.layers.DenseFeatures(feat_cols)

    # create dictionary to associate column names with column values
    inpts = {}
    for i in feat_cols:
        inpts[i.key] = tf.keras.Input(shape=(1,), name=i.key)

    # convert feat_cols to a single tensor layer
    return dset,inpts,feat_lyr

def predict_reclass_write(incloudname, model_list, threshold_vals, batch_sz, ds_cache, geo_metrics=[], geom_rad=0.10, verbose_output=2):
    '''
    Reclassify the input point cloud using the models specified in the model_list
    variable and the threshold value(s) specified in the threshold_vals list. It
    is important to note that any model using standard deviation as a model input
    should include 'sd' in the geo_metrics list.

    Input parameters:
        :param laspy.file.File incloud: Input point cloud
        :param list model_list: List of trained tensorflow models to apply.
        :param list geo_metrics: List of geometry metrics to compute.
            (NOTE: Currently limited to standard deviation - i.e. 'sd')
        :param float geom_rad: Geometric radius used to compute geometry metrics.
        :param list threshold_vals: List of threholds to use for reclassification.
            (Threshold values must be between 0.0 and 1.0)
        :param int batch_sz: Batch size for prediction/reclassification.
        :param bool ds_cache: Option to cache batches (can speed up prediction??).

    Returns:
        No values or objects are returned with this function; however,
        one or more reclassified point clouds are written as LAS files
        and subsequently converted in the function with the following format:
            '(output_filename)_(modelname)_(threshold_value)'

    Usage notes:
        If 'sdrgb' is in any of the filenames, then the function will compute
        the 3D standard deviation over the user-specified radius for every
        point in the input point cloud. Although care has been taken to speed
        up this computation process, it is still very time and resource
        intensive and should only be used when a model with standard deviation
        has been shown to be accurate for the given application. Otherwise,
        it is not recommended to submit any model using standard deviation
        to this function.

    '''
    laspy_majorversion = int(laspy.__version__.split('.')[0])
    # open both ground and vegetation files
    if laspy_majorversion == 1:
        try:
            incloud = file.File(incloudname,mode='r')
        except Exception as e:
            sys.exit(e)
    elif laspy_majorversion == 2:
        try:
            incloud = laspy.read(incloudname)
        except Exception as e:
            sys.exit(e)

    # extract model names from list of model variables
    modnamelist = [str(f.name) for f in model_list]
    print('List of models for reclassification: {}'.format(modnamelist))

    # figure out what vegetation indices to compute
    indiceslist = ['x','y','z','r','g','b']
    if any('sdrgb' in m for m in modnamelist):
        indiceslist.extend('sd')
    if any('all' in m for m in modnamelist):
        indiceslist.extend('all')
    if any('simple' in m for m in modnamelist):
        indiceslist.extend('simple')
    if any('rgb' in m for m in modnamelist):
        indiceslist.extend('rgb')

    # compute vegetation indices
    indfnames,indf = vegidx(incloud, geom_metrics=geo_metrics, indices=indiceslist, geom_radius=geom_rad)

    indat = pd.DataFrame(indf.astype('float32').transpose(), columns=indfnames)

    # FIRST, get only the columns needed for model inputs (will be matched against the model name)
    # SECOND, convert the dataframe to a tf.dataset
    if any('sdrgb' in m for m in modnamelist):
        sdrgb_df = indat[['r','g','b','sd']]
        sdrgb_ds = df_to_dataset(sdrgb_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(sdrgb_df)
    if any('xyzrgb' in m for m in modnamelist):
        xyzrgb_df = indat[['x','y','z','r','g','b']]
        xyzrgb_ds = df_to_dataset(xyzrgb_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(xyzrgb_df)
    if any('simple' in m for m in modnamelist):
        rgb_simple_df = indat[['r','g','b','exr','exg','exb','exgr']]
        rgb_simple_ds = df_to_dataset(rgb_simple_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(rgb_simple_df)
    if any('all' in m for m in modnamelist):
        all_ds = df_to_dataset(indat, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
    if any('rgb' in m for m in modnamelist):
        rgb_df = indat[['r','g','b']]
        rgb_ds = df_to_dataset(rgb_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(rgb_df)

    del(indat)
    ofname = os.path.split(incloudname)[1].split('.')[0]
    print('Output file base name: {}'.format(ofname))

    rdate = str(date.today()).replace('-','')
    if not os.path.isdir('results_'+rdate):
        os.makedirs('results_'+rdate)

    # uncomment the following block of code for use with multiple models as a list of inputs
    for m in model_list:
        print(m.name)  # print model name to console

        # predict classification
        print('Reclassifying using {} model'.format(m))
        if "simple" in m.name:
            outdat_pred = m.predict(rgb_simple_ds, batch_size=batch_sz, verbose=verbose_output, use_multiprocessing=True)
        elif "all" in m.name:
            outdat_pred = m.predict(all_ds, batch_size=batch_sz, verbose=verbose_output, use_multiprocessing=True)
        elif "sdrgb" in m.name:
            outdat_pred = m.predict(sdrgb_ds, batch_size=batch_sz, verbose=verbose_output, use_multiprocessing=True)
        elif "xyzrgb" in m.name:
            outdat_pred = m.predict(xyzrgb_ds, batch_size=batch_sz, verbose=verbose_output, use_multiprocessing=True)
        else:
            outdat_pred = m.predict(rgb_ds, batch_size=batch_sz, verbose=verbose_output, use_multiprocessing=True)

        for threshold_val in threshold_vals:
            outdat_pred_reclass = outdat_pred
            outdat_pred_reclass[(outdat_pred_reclass >= threshold_val)] = 4  # reclass veg. points
            outdat_pred_reclass[(outdat_pred_reclass < threshold_val)] = 2   # reclass no veg. points
            outdat_pred_reclass = outdat_pred_reclass.flatten().astype(np.int32)  # convert float to int

            # open the output file (laspy version dependent)
            try:
                if int(laspy.__version__.split('.')[0]) == 1:
                    print('Writing LAS file: {}'.format('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','')))
                    outfile = file.File(('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','') + '.las'), mode='w', header=incloud.header)
                elif int(laspy.__version__.split('.')[0]) == 2:
                    outfile = laspy.LasData(header=incloud.header)
            except Exception as e:
                print(e)

            # copy the points from the original file
            outfile.points = incloud.points

            # update the classification values
            outfile.classification = outdat_pred_reclass
            if int(laspy.__version__.split('.')[0]) == 1:
                outfile.close()
                print('  --> Converting from LAS to LAZ format')
                # the following functions use the subprocess module to call commands outside of Python.
                # use lastools outside of program to convert las to laz file
                subprocess.call(['las2las',
                                '-i', ('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','') + '.las'),
                                '-o', ('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','') + '.laz')])
                # remove las output file (after compressed to laz)
                subprocess.call(['rm',
                                ('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','') + '.las')])
            elif int(laspy.__version__.split('.')[0]) == 2:
                print('Writing LAZ file: {}'.format('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','')))
                outfile.write(('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','') + '.laz'))
