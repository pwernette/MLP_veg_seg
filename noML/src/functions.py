# basic libraries
import os, sys, time, ntpath
from copy import deepcopy

# import laspy (and check version)
import laspy
if int(laspy.__version__.split('.')[0]) == 1:
    try:
        from laspy import file
    except Exception as e:
        sys.exit(e)

# import libraries for managing data
import numpy as np

# import data frame libraries
import pandas as pd

# message box and file selection libraries
import tkinter
from tkinter import Tk
from tkinter.filedialog import askopenfile

from .veg_indices import *
from .dat_norm_and_format import *

# load autoreload
if '__IPYTHON__' in globals():
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext autoreload')
    ipython.magic('aimport .veg_indices')
    ipython.magic('aimport .dat_norm_and_format')
    ipython.magic('autoreload 1')

'''
Function to open a dialog window where the user can select a single file.
'''
def getfile(window_title='Select File'):
    root_win = Tk()  # initialize the tKinter window to get user input
    root_win.withdraw()
    root_win.update()
    file_io = askopenfile(title=window_title)  # get user input for the output directory
    root_win.destroy()  # destroy the tKinter window
    if not os.name=='nt':  # if the OS is not Windows, then correct slashes in directory name
        file.io = ntpath.normpath(file_io)
    return str(file_io.name)

'''
Otsu's thresholding function

Function is adapted from the following reference:
    Otsu, N. A threshold selection method from gray level histogram. IEEE Trans. Syst. Man Cybern. 1979, 9, 66–166.

The source for the code block is:
    https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python
     NOTE: The original answer does not include intensity. However, the follow-up answer (see the second
     code block) DOES include intensity in the computation of a threshold.
     Since we are trying to adapt this approach to only work with vegetation indices, we do not need to
     include the recorded intensity when thresholding.

PURPOSE:
    Function to read in an array (combined from two training class histograms) and find a threshold value that
    maximizes the interclass variability.

RETURN:
    Returns a single threshold value.
'''
def otsu_getthresh(inarr, rmin, rmax, nbins):
    num_pts = len(inarr)
    mean_weigth = 1.0/num_pts
    his,bins = np.histogram(inarr, np.arange(rmin.astype(np.float32), rmax.astype(np.float32), (rmax.astype(np.float32)-rmin.astype(np.float32))/nbins))
    final_thresh = -99999
    final_value = -99999
    for t in range(1,len(bins)-1):
        # print('bin {}'.format(t))
        # print('  his[:t] = {}'.format(his[:t]))
        # print('  his[t:] = {}'.format(his[t:]))
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth
        # print('  Wb = {}'.format(Wb))
        # print('  Wf = {}'.format(Wf))
        mub = np.mean(his[:t])
        muf = np.mean(his[t:])
        # print('  mub = {}'.format(muf))
        # print('  muf = {}'.format(muf))
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = bins[t]
            final_value = value
    # print('Final Threshold Returned = {}'.format(final_thresh))
    return final_thresh

'''
PURPOSE:
    Combine two input arrays without changing their values. This function utilizes pandas to combine the
    arrays without changing any array values.

RETURN:
    Returns a single combined numpy array containing the two input numpy arrays.

NOTES:
    Using pandas DataFrames improves performance over solely using numpy arrays:
        mergehist() with Pandas concat() (i.e. first function):       10 loops, best of 3: 19.8 ms per loop
        mergehist() with Numpy concatenate() (i.e. second function):  100 loops, best of 3: 19.4 ms per loop
'''
def mergehist(vegarr, novegarr):
    df = pd.DataFrame(vegarr)  # Two normal distributions
    df2 = pd.DataFrame(novegarr)
    df_merged = pd.concat([df, df2], ignore_index=True)
    return np.array(df_merged.values)

def scale_dims(las_file):
    '''
    Scale the X, Y, and Z coordinates within the input LAS/LAZ
    file using the offset and scaling values in the file header.

    Although this can be directly accessed with the lowercase
    'x', 'y', or 'z', it is preferrable that the integer values
    natively stored be used to avoid any loss of precision that
    may be caused by rounding.

    Returns a new data
    '''
    lpvers = int(laspy.__version__.split('.')[0])
    # create an empty numpy array to store converted values
    outdat = np.empty(shape=(0,len(las_file.X)))
    # SCALE X dimension
    x_dimension = las_file.X
    if lpvers == 1:
        try:
            scale = las_file.header.scale[0]
            offset = las_file.header.offset[0]
        except Exception as g:
            sys.exit(g)
    elif lpvers == 2:
        try:
            scale = las_file.header.scales[0]
            offset = las_file.header.offsets[0]
        except Exception as g:
            sys.exit(g)
    newrow = x_dimension*scale + offset
    outdat = np.append(outdat, np.array([newrow]), axis=0)
    # SCALE Y dimension
    y_dimension = las_file.Y
    if lpvers == 1:
        try:
            scale = las_file.header.scale[1]
            offset = las_file.header.offset[1]
        except Exception as g:
            sys.exit(g)
    elif lpvers == 2:
        try:
            scale = las_file.header.scales[1]
            offset = las_file.header.offsets[1]
        except Exception as g:
            sys.exit(g)
    newrow = y_dimension*scale + offset
    outdat = np.append(outdat, np.array([newrow]), axis=0)
    # SCALE Z dimension
    z_dimension = las_file.Z
    if lpvers == 1:
        try:
            scale = las_file.header.scale[2]
            offset = las_file.header.offset[2]
        except Exception as g:
            sys.exit(g)
    elif lpvers == 2:
        try:
            scale = las_file.header.scales[2]
            offset = las_file.header.offsets[2]
        except Exception as g:
            sys.exit(g)
    newrow = z_dimension*scale + offset
    scaledout = np.append(outdat, np.array([newrow]), axis=0)
    return scaledout

def addstats(intable, values_column='vals'):
    '''
    PURPOSE:
        Compute and append summary statistics to the vegetation indices DataFrame.

    WARNING: This function modifies the specified 'intable'
    '''
    minar,maxar,medar,meanar,stdar = [],[],[],[],[]
    for index,row in intable.iterrows():
        minar.append(np.amin(row[values_column]))
        maxar.append(np.amax(row[values_column]))
        medar.append(np.median(row[values_column]))
        meanar.append(np.mean(row[values_column]))
        stdar.append(np.std(row[values_column]))
    # outtable = pd.DataFrame(0.00, index=list(intable.index))
    intable['min'] = minar
    intable['max'] = maxar
    intable['med'] = medar
    intable['mean'] = meanar
    intable['std'] = stdar
    del([minar,maxar,medar,meanar,stdar])

def otsu(indat1, indat2):
    outdat = pd.DataFrame(0.00, index=list(indat1.index),columns=['minpossiblevalue','maxpossiblevalue','M-statistic','threshold'])
    for idx in list(indat1.index):
        outdat['minpossiblevalue'][idx] = indat1['minidxpos'][idx]
        outdat['maxpossiblevalue'][idx] = indat1['maxidxpos'][idx]
        '''
        Compute the M-statistic to compare all rows in the input table.
        From:
            Kaufman, Y.J.; Remer, L.A. Detection of forests using mid-IR reflectance: An application for aerosol studies.
            IEEE Trans. Geosci. Remote Sens. 1994, 32, 672–683.

        The vegetation index with the greatest value, should provide the greatest ability to differentiate the vegetation and non-vegetation classes.

        IMPORTANT:
            This function assumes that intable1 and intable2 have the same vegetation indices computed and reported in the same order. If the vegidx()
            function was used to compute the vegetation indices then this assumption should be valid.
        '''
        outdat['M-statistic'][idx] = (indat1['mean'][idx]-indat2['mean'][idx])/(indat1['std'][idx]-indat2['std'][idx])
        # Combine vegetation index values from vegetation and non-vegetation classes
        comboidx = mergehist(indat1['vals'][idx], indat2['vals'][idx])
        # Compute the appropriate binarization threshold value for veg/no-veg samples using Otsu's thresholding method
        # function otsu_getthresh().
        outdat['threshold'][idx] = otsu_getthresh(comboidx, outdat['minpossiblevalue'][idx], outdat['maxpossiblevalue'][idx], nbins=1000)
    return outdat

def apply_otsu(inlas, veg_index, otsu_threshold, reclasses=[2,4]):
    '''
    PURPOSE:
        Compute the appropriate vegetation index and reclassify a point cloud using previously extracted thresholds.
    '''
    # Normalize the r,g,b values
    print('Normalizing R, G, B bands...')
    inr,ing,inb = normBands(deepcopy(inlas.red), deepcopy(inlas.red), deepcopy(inlas.red))
    # compute vegetation indices based on point cloud normalized and scaled rgb values
    print('Computing vegetation indices...')
    idxvals = vegidx(inlas, veg_index)
    # create a deep copy of the original point classification values
    final_pts = deepcopy(inlas.classification)
    print('Reclassifying input point cloud using "{}" with threshold of {}'.format(veg_index, otsu_threshold))
    print('  Original Classes: {}'.format(final_pts))
    # reclassify the original classification values based on the Otsu threshold and reclassification scheme specified
    final_pts[idxvals['vals'][veg_index] > otsu_threshold] = reclasses[1]
    final_pts[idxvals['vals'][veg_index] < otsu_threshold] = reclasses[0]
    print('   UPDATED Classes: {}'.format(final_pts))
    # return the updated classificiation values as an array of numpy integers
    return final_pts
