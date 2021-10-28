import sys
import subprocess
import pandas as pd
import numpy as np

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

# load custom modules
from .vegindex import vegidx
from .miscfx import *
from .modelbuilder import *

def las2split(infile_ground_pc, infile_veg_pc, veg_indices=[],
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
    names_ground,dat_ground = vegidx(fground, indices=veg_indices)
    names_veg,dat_veg = vegidx(fveg, indices=veg_indices)

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
def df_to_dataset(dataframe, targetcolname='none', shuffle=False, prefetch=False, cache_ds=False, batch_size=32):
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
    if not targetcolname == 'none':
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


def pd2fl(input_pd_dat, col_names=['r','g','b'], targetcol='veglab', dat_type='float32', shuf=False, batch_sz=32, ds_prefetch=False, ds_cache=False, verbose=True):
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
    if col_names=='all':
        col_names = list(input_pd_dat.columns)
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

def predict_reclass_write(incloud, ofname, model_list, geo_metrics=[], geom_rad=0.10, threshold_vals=[0.5], batch_sz=1000, ds_cache=False, col_depth=16):
    '''
    Reclassify the input point cloud using the models specified in the
    model_list variable and the threshold value(s) specified in the
    threshold_vals list. It is important to note that any model using standard
    deviation as a model input should include 'sd' in the geo_metrics list.

    Input parameters:
        :param laspy.file.File incloud: Input point cloud
        :param string ofname: Output file name
        :param list model_list: List of trained tensorflow models to apply.
        :param list geo_metrics: List of geometry metrics to compute.
            (NOTE: Currently limited to standard deviation - i.e. 'sd')
        :param float geom_rad: Geometric radius to compute geometry metrics.
        :param list threshold_vals: List of threholds for reclassification.
            (Threshold values must be between 0.0 and 1.0)
        :param int batch_sz: Batch size for prediction/reclassification.
        :param bool ds_cache: Option to cache batches (can speed up prediction).

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
            f = file.File(incloud,mode='r')
        except Exception as e:
            sys.exit(e)
    elif laspy_majorversion == 2:
        try:
            f  =laspy.read(incloud)
        except Exception as e:
            sys.exit(e)

    in_ds,in_cols,in_lyr = pd2fl(col_names=['r','g','b'], dat_type='float32', shuf=False, batch_sz=32, ds_prefetch=True, verbose=True))
    # compute vegetation indices
    colnames,indat = vegidx(incloud, indices=veg_indices)

    # transpose the output objects
    dat_sample = np.transpose(indat)

    # OPTIONAL: print number of points in each input dense point cloud
    if verbose:
        print('# of points     = {}'.format(ground_sample.shape))

    if len(threshold_vals) == 0:
        print('WARNING: No threshold values were specified. Using a default threshold value of 0.5.')
        threshold_vals = [0.5]

    # compute vegetation indices
    if any('sd' in m for m in model_list):
        if any('all' in m for m in model_list):
            indfnames,indf = vegidx(incloud, geom_metrics=geo_metrics, indices=['all','sd'], geom_radius=geom_rad)
        elif any('simple' in m for m in model_list):
            indfnames,indf = vegidx(incloud, geom_metrics=geo_metrics, indices=['simple','sd'], geom_radius=geom_rad)
        else:
            indfnames,indf = vegidx(incloud, geom_metrics=geo_metrics, indices=['sd'], geom_radius=geom_rad)
    else:
        if any('all' in m for m in model_list):
            indfnames,indf = vegidx(incloud, indices=['all'])
        elif any('simple' in m for m in model_list):
            indfnames,indf = vegidx(incloud, indices=['simple'])
        else:
            indfnames,indf = vegidx(incloud, indices=[])

    indat = pd.DataFrame(indf.astype('float32').transpose(), columns=indfnames)

    # FIRST, get only the columns needed for model inputs (will be matched against the model name)
    # SECOND, convert the dataframe to a tf.dataset
    if any('sdrgb' in m for m in model_list):
        sdrgb_df = indat[['r','g','b','sd']]
        sdrgb_ds = df_to_dataset(sdrgb_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(sdrgb_df)
    if any('xyzrgb' in m for m in model_list):
        xyzrgb_df = indat[['x','y','z','r','g','b']]
        xyzrgb_ds = df_to_dataset(xyzrgb_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(xyzrgb_df)
    if any('simple' in m for m in model_list):
        rgb_simple_df = indat[['r','g','b','exr','exg','exb','exgr']]
        rgb_simple_ds = df_to_dataset(rgb_simple_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(rgb_simple_df)
    if any('all' in m for m in model_list):
        all_ds = df_to_dataset(indat, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
    if any('rgb_8' in m for m in model_list) or any('rgb_16' in m for m in model_names_list):
        rgb_df = indat[['r','g','b']]
        rgb_ds = df_to_dataset(rgb_df, shuffle=False, cache_ds=ds_cache, batch_size=batch_sz)
        del(rgb_df)

    del(indat)

    for m in model_list:
        print(m)  # print model name to console
#         globals()[m]
        mod = globals()[(m)]

        # predict classification
        if "simple" in m:
            outdat_pred = mod.predict(rgb_simple_ds, batch_size=batch_sz, verbose=1, use_multiprocessing=True)
        elif "all" in m:
            outdat_pred = mod.predict(all_ds, batch_size=batch_sz, verbose=1, use_multiprocessing=True)
        elif "sdrgb" in m:
            outdat_pred = mod.predict(sdrgb_ds, batch_size=batch_sz, verbose=1, use_multiprocessing=True)
        elif "xyzrgb" in m:
            outdat_pred = mod.predict(xyzrgb_ds, batch_size=batch_sz, verbose=1, use_multiprocessing=True)
        else:
            outdat_pred = mod.predict(rgb_ds, batch_size=batch_sz, verbose=1, use_multiprocessing=True)

        for threshold_val in threshold_vals:
            outdat_pred_reclass = outdat_pred
            outdat_pred_reclass[(outdat_pred_reclass >= threshold_val)] = 4  # reclass veg. points
            outdat_pred_reclass[(outdat_pred_reclass < threshold_val)] = 2   # reclass no veg. points
            outdat_pred_reclass = outdat_pred_reclass.flatten().astype(np.int32)  # convert float to int

            if int(laspy.__version__.split('.')[0]) == 1:
                outfile = file.File((ofname + "_" + str(m) + "_" + str(threshold_val) + '.las'), mode='w', header=incloud.header)
            elif int(laspy.__version__.split('.')[0]) == 2:
                outfile = laspy.LasData(header=incloud.header)
            except Exception as e:
                print(e)

            # copy the points information from the original file
            outfile.points = incloud.points

            # update the classification values
            outfile.classification = outdat_pred_reclass
            if int(laspy.__version__.split('.')[0]) == 1:
                outfile.close()
            elif int(laspy.__version__.split('.')[0]) == 2:
                outfile.write((ofname + "_" + str(m) + "_" + str(threshold_val) + '.las'))

            # the following functions use the subprocess module to call commands outside of Python.
            # use lastools outside of program to convert las to laz file
            subprocess.call(['las2las',
                            '-i', (ofname + "_" + str(m) + "_" + str(threshold_val) + '.las'),
                            '-o', (ofname + "_" + str(m) + "_" + str(threshold_val) + '.laz')])
            # remove las output file (after compressed to laz)
            subprocess.call(['rm',
                            (ofname + "_" + str(m) + "_" + str(threshold_val) + '.las')])
