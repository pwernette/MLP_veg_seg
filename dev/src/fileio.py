import sys
import laspy
if int(laspy.__version__.split('.')[0]) == 1:
    from laspy import file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from vegindex import vegidx
import miscfx

def las2split(infile_ground_pc, infile_veg_pc, veg_indices=[], training_split=0.7, class_imbalance_corr=True, data_reduction=1.0, verbose=True):
    '''
    (1) Read LAS/LAZ files representing the ground and vegetation training
    points, (2) resample the larger file to the same size as the smaller file
    (addresses class imbalance), (3) combine the two point clouds to a single
    pandas.DataFrame and append a new column representing ground-veg dichotomy,
    and (4) perform a train-test-validation split of the data.
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
    names_ground,dat_ground = vegindex.vegidx(fground, indices=veg_indices)
    names_veg,dat_veg = vegindex.vegidx(fveg, indices=veg_indices)

    # transpose the output objects
    ground_sample  =np.transpose(dat_ground)
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

    # these lines sub-sample the vegetation and no-vegetation data to cut the data volume
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

    # OPTIONAL: print info about training, testing, validation split numbers
    if verbose:
        print('  {} train examples'.format(len(trainout)))
        print('  {} validation examples'.format(len(valout)))
        print('  {} test examples'.format(len(testout)))

    # return the train, test, and validationn objects
    # all outputs are pandas.DataFrame objects
    return trainout,testout,valout


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, targetcolname='none', shuffle=True, prefetch=False, cache_ds=False, batch_size=32):
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
        ds = ds.prefetch(tf.data.AUTOTUNE)
    if cache_ds:
        ds = ds.cache()
    return ds


def pd2fl(input_pd_dat, col_names=['r','g','b'], targetcol='veglab', dat_type='float32', shuf=False, batch_sz=1000, verbose=True):
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
    :param bool verbose: Optional argument to print information to console

    Returns:

        * dset (:py:class:`tf.Dataset`)
        * inpts (:py:class:`list`)
        * lyr (:py:class:`tf.DenseFeatures`)
    '''
    dset = df_to_dataset(input_pd_dat[col_names].astype(dat_type),
                                 targetcolname=targetcol,
                                 shuffle=shuf,
                                 batch_size=batch_sz)
    feat_cols = []
    for header in col_names:  # using only columns 3-5 will limit the table to RGB only
        feat_cols.append(feature_column.numeric_column(header))
    if verbose:
        for i in feat_cols:
            print(i)

    # create dense features layer
    feat_lyr = tf.keras.layers.DenseFeatures(feat_cols)

    # create dictionary to associate column names with column values
    inpts = {}
    for i in feat_cols:
        inpts[i.key] = tf.keras.Input(shape=(1,), name=i.key)

    lyr = feat_lyr(inpts)
#     del(feat_cols, feat_lyr, inpts)
#     features = tf.io.parse_example(
#         features=tf.feature_column.make_parse_example_spec(feat_cols))

    # convert feat_cols to a single tensor layer
    return dset,inpts,lyr
