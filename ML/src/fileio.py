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
                veg_indices=[], 
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
    dat_list = []
    min_pts = 999999999999999

    dat_dict = {}
    class_val = 0

    # get minimum number of points
    for ifile in infile_pcs:
        # open the file to access only the header
        inhead = laspy.open(ifile)
        # get the minimum number of points in all input point counts
        if inhead.header.point_count < min_pts:
            min_pts = inhead.header.point_count

    # try:
    for ifile in infile_pcs:
        # print(ifile)
        input_files.append(ifile)
        if laspy_majorversion == 1:
            indat = file.File(ifile, mode='r')
        elif laspy_majorversion == 2:
            indat = laspy.read(ifile)
        print('Read {} using laspy major version: {}'.format(ifile, laspy_majorversion))

        # compute vegetation indices and generate dataframe
        if veg_indices == 'rgb' or veg_indices is None:
            indat = veg_rgb(indat)
        else:
            print('\nGeometry metrics specified: {}'.format(geometry_metrics))
            indat = vegidx(indat, 
                           indices=veg_indices, 
                           geom_metrics=geometry_metrics)
        
        # names_list.append(os.path.splitext(os.path.basename(ifile))[0]+'_names')
        # dat_list.append(os.path.splitext(os.path.basename(ifile))[0]+'_dat')

        # convert the samples to Pandas DataFrame objects
        indat = generate_dataframe(indat, 
                                   veg_indices,
                                   dtype_conversion='float32')
        indat['veglab'] = np.full(shape=len(indat), 
                                  fill_value=class_val, 
                                  dtype=np.float32)
        # print(indat['veglab'])

        # # transpose the data
        # globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'] = np.transpose(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'])
        # convert the data to pandas DataFrame
        # globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'] = generate_dataframe(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_dat'], 
        #                                                                                     veg_indices, 
        #                                                                                     dtype_conversion='float32')

        # # append the names with the vegetation label column name
        # globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'] = np.append(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'], 'veglab')
        
        # print(globals()[os.path.splitext(os.path.basename(ifile))[0]+'_names'])
        # if laspy_majorversion == 1:
        #     inlas.close()
        #     print('\n\nERROR: Unable to close {}\n\n'.format(inlas))
        # del(inlas)

        # sample larger dat to match size of smaller dat
        if class_imbalance_corr:
            if len(indat) > min_pts:
                indat = train_test_split(indat, train_size=min_pts/len(indat), random_state=42)[0]
                print('Class Imbalance Correction: Randomly sampled {} to {} points'.format(ifile, len(indat)))
        
        # sub-sample the data to cut the data volume
        if data_reduction < 1.0:
            indat = train_test_split(indat, train_size=data_reduction, random_state=42)[0]
            print('Data Reduction: Randomly sampled {} to {} points'.format(ifile, len(indat)))

        # write dictionary of data name and corresponding numerical value
        dat_dict[os.path.basename(ifile)] = class_val

        print('\nSplitting {}:'.format(ifile))

        # split the data to training, validation, and evaluation
        traind,evald = train_test_split(indat, train_size=training_split, random_state=42)
        traind,vald = train_test_split(traind, train_size=training_split, random_state=42)

        # concatenate new training data to dataframe
        if class_val == 0:
            trainout = traind
        else:
            trainout = pd.concat([trainout,traind], ignore_index=True)
        
        # concatenate new validation data to dataframe
        if class_val == 0:
            valout = vald
        else:
            valout = pd.concat([valout,vald], ignore_index=True)
        
        # concatenate new evaluation data to dataframe
        if class_val == 0:
            evalout = evald
        else:
            evalout = pd.concat([evalout,evald], ignore_index=True)

        if verbose:
            print('    {} training points'.format(len(traind)))
            print('    {} validation points'.format(len(vald)))
            print('    {} evaluation points'.format(len(evald)))

        # clean up memory
        del(traind,vald,evald)

        class_val += 1

    # return the train, validation, and evaluation objects
    # all outputs are pandas.DataFrame objects WITH an additional veglab attribute
    return trainout.sample(frac=1), valout.sample(frac=1), evalout.sample(frac=1), dat_dict


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, 
                  targetcolname='none', 
                  label_depth=2, 
                  shuffle=True, 
                  prefetch=False, 
                  cache_ds=False, 
                  batch_size=32, 
                  drop_remain=True,
                  verbose=0):
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
        if verbose == 2:
            print(dataframe.head())
    if not targetcolname == 'none':
        # assume that the last column is the labels
        # labels = np.array(dataframe.pop(dataframe.columns[-1]))
        labels = np.array(dataframe.pop(targetcolname))
        
        # ds = tf.data.Dataset.from_tensor_slices((ds_inputs, labels))
        # ds = tf.data.Dataset.from_tensor_slices((ds_inputs, tf.one_hot(labels, depth=label_depth)))
        # # ds = tf.data.Dataset.from_tensor_slices((ds_inputs, tf.keras.utils.to_categorical(labels)))
        # if verbose > 0:
        #     print(ds)
        # ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        # if cache_ds:
        #     ds = ds.cache()
        # if prefetch:
        #     ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        labels = tf.one_hot(labels, depth=label_depth)
        ds_inputs = tf.convert_to_tensor(dataframe)
        ds = tf.data.Dataset.from_tensor_slices((ds_inputs, labels))

        if verbose == 2:
            print(ds)
            print(ds.element_spec[0].shape[1:])
    else:
        ds_inputs = tf.convert_to_tensor(dataframe)
        ds = tf.data.Dataset.from_tensor_slices((ds_inputs))

        if verbose == 2:
            print(ds)

    # if shuffle:
    #     ds = ds.shuffle(buffer_size=batch_size)
    ds = ds.batch(batch_size, drop_remainder=drop_remain, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_ds:
        ds = ds.cache()
    if prefetch:
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return ds

# lookup_dict = {
#     'simple': ['exr','exg','exb','exgr'],
#     'coords': ['x','y','z'],
#     'sd': ['sd'],
#     'all': ['ngrdi','mgrvi','gli','rgbvi','ikaw','gla']
# }
def generate_dataframe(input_point_cloud, vegetation_index_list, dtype_conversion='float32'):
    outdict = {}
    outdict['r'] = np.array(input_point_cloud.rnorm, dtype=dtype_conversion).flatten()
    outdict['g'] = np.array(input_point_cloud.gnorm, dtype=dtype_conversion).flatten()
    outdict['b'] = np.array(input_point_cloud.bnorm, dtype=dtype_conversion).flatten()
    if any('xyz' in m for m in vegetation_index_list):
        outdict['x'] = np.array(input_point_cloud.x, dtype=dtype_conversion).flatten()
        outdict['y'] = np.array(input_point_cloud.y, dtype=dtype_conversion).flatten()
        outdict['z'] = np.array(input_point_cloud.z, dtype=dtype_conversion).flatten()
    if any('3d' in m for m in vegetation_index_list):
        outdict['sd3d'] = np.array(input_point_cloud.sd3d, dtype=dtype_conversion).flatten()
    if any('sd' in m for m in vegetation_index_list):
        outdict['sd_x'] = np.array(input_point_cloud.sd_x, dtype=dtype_conversion).flatten()
    if any('exg' == m for m in vegetation_index_list) or any('simple' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['exg'] = np.array(input_point_cloud.exg, dtype=dtype_conversion).flatten()
    if any('exr' in m for m in vegetation_index_list) or any('simple' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['exr'] = np.array(input_point_cloud.exr, dtype=dtype_conversion).flatten()
    if any('exb' in m for m in vegetation_index_list) or any('simple' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['exb'] = np.array(input_point_cloud.exb, dtype=dtype_conversion).flatten()
    if any('exgr' in m for m in vegetation_index_list) or any('simple' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['exgr'] = np.array(input_point_cloud.exgr, dtype=dtype_conversion).flatten()
    if any('ngrdi' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['ngrdi'] = np.array(input_point_cloud.ngrdi, dtype=dtype_conversion).flatten()
    if any('mgrvi' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['mgrvi'] = np.array(input_point_cloud.mgrvi, dtype=dtype_conversion).flatten()
    if any('gli' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['gli'] = np.array(input_point_cloud.gli, dtype=dtype_conversion).flatten()
    if any('rgbvi' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['rgbvi'] = np.array(input_point_cloud.rgbvi, dtype=dtype_conversion).flatten()
    if any('ikaw' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['ikaw'] = np.array(input_point_cloud.ikaw, dtype=dtype_conversion).flatten()
    if any('gla' in m for m in vegetation_index_list) or any('all' in m for m in vegetation_index_list):
        outdict['gla'] = np.array(input_point_cloud.gla, dtype=dtype_conversion).flatten()
    
    return pd.DataFrame(outdict)

def predict_reclass_write(incloudname, model_list, threshold_vals, batch_sz=32, ds_cache=False, indiceslist=[], geo_metrics=[], geom_rad=0.10, verbose_output=1):
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

    if not isinstance(indiceslist, list):
        indiceslist = list(indiceslist)

    # figure out what vegetation indices to compute
    if any('xyz' in m for m in modnamelist):
        indiceslist.append('xyz')
    if any('3d' in m for m in modnamelist):
        indiceslist.append('3d')
    if any('sd' in m for m in modnamelist):
        indiceslist.append('sd')
    if any('all' in m for m in modnamelist):
        indiceslist = 'all'
    if any('simple' in m for m in modnamelist):
        indiceslist = 'simple'
    if any('exr' in m for m in modnamelist):
        indiceslist.append('exr')
    if any('exg' in m for m in modnamelist) and not any('exgr' in m for m in modnamelist):
        indiceslist.append('exg')
    if any('exgr' in m for m in modnamelist):
        indiceslist.append('exgr')
    if any('exb' in m for m in modnamelist):
        indiceslist.append('exb')
    if any('ngrdi' in m for m in modnamelist):
        indiceslist.append('ngrdi')
    if any('mgrvi' in m for m in modnamelist):
        indiceslist.append('mgrvi')
    if any('gli' in m for m in modnamelist):
        indiceslist.append('gli')
    if any('rgbvi' in m for m in modnamelist):
        indiceslist.append('rgbvi')
    if any('ikaw' in m for m in modnamelist):
        indiceslist.append('ikaw')
    if any('gla' in m for m in modnamelist):
        indiceslist.append('gla')

    # compute vegetation indices
    incloud = vegidx(incloud, 
                     geom_metrics=geo_metrics, 
                     indices=indiceslist, 
                     geom_radius=geom_rad)

    # generate Pandas DataFrame from computed indices in the point cloud
    indat = generate_dataframe(incloud, 
                               indiceslist, 
                               dtype_conversion='float32')
    print(indat)
    
    # convert the dataframe to a TF dataset
    converted_dataset = df_to_dataset(indat, 
                                      targetcolname='none', 
                                      shuffle=False, 
                                      cache_ds=ds_cache, 
                                      batch_size=batch_sz,
                                      drop_remain=False)
    # print(converted_dataset)
    # print(len(converted_dataset))

    del(indat)
    odir,ofname = os.path.split(incloudname)
    ofname = ofname.split('.')[0]
    print('Output file base name: {}'.format(ofname))

    rdate = str(date.today()).replace('-','')
    if not os.path.isdir('results_'+rdate):
        os.makedirs('results_'+rdate)

    # uncomment the following block of code for use with multiple models as a list of inputs
    for m in model_list:
        # predict classification
        print('Reclassifying using {} model'.format(m.name))
        outdat_pred = m.predict(converted_dataset, verbose=2, use_multiprocessing=True)

        if verbose_output == 2:
            print('\nOutput Predictions (raw): {}'.format(len(outdat_pred)))
            print(outdat_pred)

        outdat_pred = tf.argmax(outdat_pred,-1)

        if verbose_output == 2:
            print('\nOutput Predictions: {}'.format(len(outdat_pred)))
            print(outdat_pred)

        # print('threshold_vals = {}'.format(threshold_vals))
        if not isinstance(threshold_vals, list):
            try:
                threshold_vals = [v for v in threshold_vals]
            except:
                try:
                    threshold_vals = list(map(float, threshold_vals))
                except:
                    print('No conversion of the threshold_values object to list took place.')
                    pass
        print('Threshold Value(s) = {}'.format(threshold_vals))
        if isinstance(threshold_vals, list):
            for threshold_val in list(threshold_vals):
                outdat_pred_reclass = outdat_pred
                # outdat_pred_reclass[(outdat_pred_reclass >= threshold_val)] = 4  # reclass veg. points
                # # outdat_pred_reclass[(outdat_pred_reclass < threshold_val)] = 2   # reclass no veg. points
                # outdat_pred_reclass = outdat_pred_reclass.flatten().astype(np.int32)  # convert float to int

                
                # open the output file (laspy version dependent)
                if int(laspy.__version__.split('.')[0]) == 1:
                    print('Writing LAS file: {}'.format(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.las')))
                    outfile = file.File((os.path.join(odir,ofname.replace('las','').replace('laz','')+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.las')), mode='w', header=incloud.header)
                    # copy the points from the original file
                    outfile.points = incloud.points
                # elif int(laspy.__version__.split('.')[0]) == 2:
                #     outfile = laspy.LasData(header=incloud.header)

                if int(laspy.__version__.split('.')[0]) == 1:
                    # update the classification values
                    outfile.classification = outdat_pred_reclass
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
                    # update the classification values
                    incloud.classification = outdat_pred_reclass
                    # write out the new file (with vegetation indices included as extra bytes, is specified)
                    print('Writing LAZ file: {}'.format(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.copc.laz')))
                    incloud.write(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_val).replace('.','')+'.copc.laz'))
        else:
            outdat_pred_reclass = outdat_pred
            # outdat_pred_reclass[(outdat_pred_reclass >= threshold_vals)] = 4  # reclass veg. points
            # # outdat_pred_reclass[(outdat_pred_reclass < threshold_vals)] = 2   # reclass no veg. points
            # outdat_pred_reclass = outdat_pred_reclass.flatten().astype(np.int32)  # convert float to int

            # open the output file (laspy version dependent)
            if int(laspy.__version__.split('.')[0]) == 1:
                print('Writing LAS file: {}'.format(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_vals).replace('.','')+'.las')))
                outfile = file.File((os.path.join(odir,ofname.replace('las','').replace('laz','')+"_"+str(m.name)+"_"+str(threshold_vals).replace('.','')+'.las')), mode='w', header=incloud.header)
                # copy the points from the original file
                outfile.points = incloud.points
            # elif int(laspy.__version__.split('.')[0]) == 2:
            #     outfile = laspy.LasData(header=incloud.header)

            if int(laspy.__version__.split('.')[0]) == 1:
                # update the classification values
                outfile.classification = outdat_pred_reclass
                outfile.close()
                print('  --> Converting from LAS to LAZ format')
                # the following functions use the subprocess module to call commands outside of Python.
                # use lastools outside of program to convert las to laz file
                subprocess.call(['las2las',
                                '-i', (os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_vals).replace('.','')+'.las')),
                                '-o', (os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_vals).replace('.','')+'.laz'))])
                # remove las output file (after compressed to laz)
                subprocess.call(['rm',
                                ('results_' + rdate + '/' + ofname + "_" + str(m.name) + "_" + str(threshold_vals).replace('.','') + '.las')])
            elif int(laspy.__version__.split('.')[0]) == 2:
                # update the classification values
                # print('\nincloud.classification: {}'.format(len(incloud.classification)))
                # print(incloud.classification)
                # print('\nOutput Predictions: {}'.format(len(outdat_pred_reclass)))
                # print(outdat_pred_reclass)
                incloud.classification = outdat_pred_reclass
                # write out the new file (with vegetation indices included as extra bytes, is specified)
                print('Writing LAZ file: {}'.format(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_vals).replace('.','')+'.copc.laz')))
                incloud.write(os.path.join(odir,ofname+"_"+str(m.name)+"_"+str(threshold_vals).replace('.','')+'.copc.laz'))
