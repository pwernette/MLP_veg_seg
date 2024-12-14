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
from .vegindex import vegidx, veg_rgb
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

def las2split(infile_pcs,
                veg_indices=['rgb'], 
                geometry_metrics=[],
                training_split=0.7, 
                class_imbalance_corr=True,
                data_reduction=1.0, 
                verbose=True):
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

    input_files = []
    names_list = []
    dat_list = []
    min_pts = 999999999999999

    try:
        for ifile in infile_pcs:
            print(ifile)
            input_files.append(ifile)
            if laspy_majorversion == 1:
                inlas = file.File(ifile, mode='r')
            elif laspy_majorversion == 2:
                inlas = laspy.read(ifile)
            print('Read {} using laspy major version: {}'.format(ifile, laspy_majorversion))

            # compute vegetation indices
            if veg_indices=='rgb' or veg_indices is None:
                globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'], globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'] = veg_rgb(inlas)
            else:
                globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'], globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'] = vegidx(inlas, indices=veg_indices, geom_metrics=geometry_metrics)
            
            names_list.append(os.path.splitext(os.path.basename(ifile))[0]+'_names')
            dat_list.append(os.path.splitext(os.path.basename(ifile))[0]+'_dat')

            # transpose the data
            globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'] = np.transpose(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'])

            # populate the dictionary of point counts
            if globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'].shape[0] < min_pts:
                min_pts = globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'].shape[0]

            # append the names with the vegetation label column name
            globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'] = np.append(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'], 'veglab')
            
            print(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'])
            if laspy_majorversion == 1:
                inlas.close()
                print('\n\nERROR: Unable to close {}\n\n'.format(inlas))
            del(inlas)
    except Exception as e:
        sys.exit(e)

    # OPTIONAL: print number of points in each input dense point cloud
    if verbose:
        print('\nPoint cloud counts:')
        [print('{} contains {} points'.format(input_files[i], globals()[v].shape)) for i,v in enumerate(dat_list)]

    dat_dict = {}

    for i,d in enumerate(dat_list):
        print('\nSplitting {}:'.format(input_files[i]))
        # sample larger dat to match size of smaller dat
        if class_imbalance_corr:
            if globals()[d].shape[0] > min_pts:
                globals()[d] = train_test_split(globals()[d], train_size=min_pts/globals()[d].shape[0], random_state=42)[0]
        
        # sub-sample the data to cut the data volume
        if data_reduction < 1.0:
            globals()[d] = train_test_split(globals()[d], train_size=data_reduction, random_state=42)[0]
        
        # convert the samples to Pandas DataFrame objects
        globals()[d] = pd.DataFrame(globals()[d].astype('float32'), columns=globals()[names_list[i]][:-1])
        globals()[d]['veglab'] = np.full(shape=globals()[d].shape[0], fill_value=i, dtype=np.float32)

        # write dictionary of data name and corresponding numerical value
        dat_dict[input_files[i]] = i

        # split the data to training, validation, and evaluation
        traind,evald = train_test_split(globals()[d], train_size=training_split, random_state=42)
        traind,vald = train_test_split(traind, train_size=training_split, random_state=42)

        # concatenate new training data to dataframe
        if not 'trainout' in globals():
            globals()['trainout'] = traind
        else:
            globals()['trainout'] = pd.concat([globals()['trainout'],traind], ignore_index=True)
        
        # concatenate new validation data to dataframe
        if not 'valout' in globals():
            globals()['valout'] = vald
        else:
            globals()['valout'] = pd.concat([globals()['valout'],vald], ignore_index=True)
        
        # concatenate new evaluation data to dataframe
        if not 'evalout' in globals():
            globals()['evalout'] = vald
        else:
            globals()['evalout'] = pd.concat([globals()['evalout'],evald], ignore_index=True)

        if verbose:
            print('    {} training points'.format(len(traind)))
            print('    {} validation points'.format(len(vald)))
            print('    {} evaluation points'.format(len(evald)))

        # clean up memory
        del(traind,vald,evald)

    # return the train, validation, and evaluation objects
    # all outputs are pandas.DataFrame objects WITH an additional veglab attribute
    return trainout, valout, evalout, dat_dict


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, targetcolname='', shuffle=False, prefetch=False, cache_ds=False, batch_size=32, verbose=0):
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
        if verbose > 0:
            print(dataframe.head())
    if not targetcolname == '':
        labels = dataframe.pop(targetcolname)
        ds_inputs = tf.convert_to_tensor(dataframe)
        ds = tf.data.Dataset.from_tensor_slices((ds_inputs, labels))
        if verbose > 0:
            print(ds)
        ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        if cache_ds:
            ds = ds.cache()
        if prefetch:
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.convert_to_tensor(dataframe)
    if verbose > 0:
        print(ds)
    return ds

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
    indiceslist = ['r','g','b']
    if any('xyzrgb' in m for m in modnamelist):
        indiceslist.extend('xyz')
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
    odir,ofname = os.path.split(incloudname)
    ofname = ofname.split('.')[0]
    print('Output file base name: {}'.format(ofname))

    rdate = str(date.today()).replace('-','')
    if not os.path.isdir('results_'+rdate):
        os.makedirs('results_'+rdate)

    # uncomment the following block of code for use with multiple models as a list of inputs
    for m in model_list:
        print(m.name)  # print model name to console

        # predict classification
        print('Reclassifying using {} model'.format(m.name))
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

        print('threshold_vals = {}'.format(threshold_vals))
        if not isinstance(threshold_vals, list):
            threshold_vals = [threshold_vals]
        for threshold_val in threshold_vals:
            outdat_pred_reclass = outdat_pred
            outdat_pred_reclass[(outdat_pred_reclass >= threshold_val)] = 4  # reclass veg. points
            outdat_pred_reclass[(outdat_pred_reclass < threshold_val)] = 2   # reclass no veg. points
            outdat_pred_reclass = outdat_pred_reclass.flatten().astype(np.int32)  # convert float to int

            # open the output file (laspy version dependent)
            try:
                if int(laspy.__version__.split('.')[0]) == 1:
                    print('Writing LAS file: {}'.format(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.las')))
                    outfile = file.File((os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.las')), mode='w', header=incloud.header)
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
                                '-i', (os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.las')),
                                '-o', (os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.laz'))])
                # remove las output file (after compressed to laz)
                subprocess.call(['rm',
                                ('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_val).replace('.','') + '.las')])
            elif int(laspy.__version__.split('.')[0]) == 2:
                print('Writing LAZ file: {}'.format(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.laz')))
                outfile.write(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.laz'))
