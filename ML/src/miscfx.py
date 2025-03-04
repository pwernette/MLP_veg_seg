import sys, os, math
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# import machine learning libraries
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping

# message box and file selection libraries
import tkinter
from tkinter import Tk
from tkinter.filedialog import askopenfile
from tkinter import simpledialog

def arr2float32(inarr):
    '''
    Convert an array to float32.
    '''
    if inarr.dtype != np.float32:
        outarr = inarr.astype(np.float32)
    return np.asarray(outarr)
def arr2float32(inarrlist):
    '''
    Convert a 2-dimensional array to float32.
    '''
    outarrlist = []
    for inarr in inarrlist:
        if inarr.dtype != np.float32:
            outarr = inarr.astype(np.float32)
            outarrlist.append([outarr])
    return np.asarray(outarrlist)


def arr2float(inarr, targetdtype='float32'):
    '''
    Convert an array to a specific numeric type, as
    specified by the user.

    :param numpy.array inarr: Array to be converted
    :param str targetdtype: Data type to be converted to.
        'float16' --> 16-bit floating point
        'float32' --> 32-bit floating point
        'float64' --> 64-bit floating point

    Returns a new array of dtype specified.
    '''
    if targetdtype=='float32':
        if inarr.dtype != np.float32:
            outarr = inarr.astype(np.float32)
    elif targetdtype=='float64':
        if inarr.dtype != np.float64:
            outarr = inarr.astype(np.float64)
    elif targetdtype=='float16':
        if inarr.dtype != np.float16:
            outarr = inarr.astype(np.float16)
    return outarr
def arr2float(inarrlist, targetdtype='float32'):
    '''
    Convert a n-dimensional array to a specific numeric
    type, as specified by the user.

    :param numpy.array inarrlist: N-dimensional array of
        arrays to be converted
    :param str targetdtype: Data type to be converted to.
        'float16' --> 16-bit floating point
        'float32' --> 32-bit floating point
        'float64' --> 64-bit floating point

    Returns a new n-dimensional array of dtype specified.
    '''
    outarrlist = []
    for inarr in inarrlist:
        if targetdtype=='float32':
            if inarr.dtype != np.float32:
                outarr = inarr.astype(np.float32)
                outarrlist.append([outarr])
        elif targetdtype=='float64':
            if inarr.dtype != np.float64:
                outarr = inarr.astype(np.float64)
                outarrlist.append([outarr])
        elif targetdtype=='float16':
            if inarr.dtype != np.float16:
                outarr = inarr.astype(np.float16)
                outarrlist.append([outarr])
    return outarrlist


def normdat(inarr, minval=0, maxval=65535, normrange=[0,1]):
    '''
    Normalize values in the input array with the specified
    min and max values to the output range normrange[].

    :param numpy.array inarr: Array to be normalized
    :param int minval: Minimum value of the input data
    :param int maxval: Maximum value of the input data
    :param tuple normrange: Range that values should be
        re-scaled to (default = 0 to 1)

    Returns a new array with normalized values.
    '''
    if minval!=0:
        minval = np.amin(inarr)
    if maxval!=65535:
        maxval = np.amax(inarr)
    norminarr = (normrange[1]-normrange[0])*np.divide((inarr-np.asarray(minval)), (np.asarray(maxval)-np.asarray(minval)), out=np.zeros_like(inarr), where=(maxval-minval)!=0)-normrange[0]
    return norminarr


def normBands(b1, b2, b3, depth=16):
    '''
    Normalize all bands in a 3-band input.

    :param numpy.array b1: Input band 1
    :param numpy.array b2: Input band 2
    :param numpy.array b3: Input band 3
    :param int depth: Bit-depth of the input data
        (default is 16-bit data, which is limited to 0,65535)

    Returns three normalized bands:

        * b1normalized (:py:class:`float`)
        * b2normalized (:py:class:`float`)
        * b3normalized (:py:class:`float`)
    '''
    # ensure that bands are converted to 32-bit float
    b1,b2,b3 = arr2float([b1,b2,b3], targetdtype='float32')
    # check band depth
    if depth==16:
        b1min,b1max,b2min,b2max,b3min,b3max = 0,65535,0,65535,0,65535
    elif depth==8:
        b1min,b1max,b2min,b2max,b3min,b3max = 0,255,0,255,0,255
    else:
        sys.exit("ERROR: bit-depth must be 8 or 16.")
    # normalize bands indivudally first
    b1norm = normdat(b1, minval=b1min, maxval=b1max)
    b2norm = normdat(b2, minval=b2min, maxval=b2max)
    b3norm = normdat(b3, minval=b3min, maxval=b3max)
#     print(b1norm)  # FOR DEBUGGING
    # then normalize one band by all others
    b1normalized = np.divide(b1norm, (b1norm+b2norm+b3norm),
                        out=np.zeros_like(b1norm),
                        where=(b1norm+b2norm+b3norm)!=0)
    b2normalized = np.divide(b2norm, (b1norm+b2norm+b3norm),
                        out=np.zeros_like(b2norm),
                        where=(b1norm+b2norm+b3norm)!=0)
    b3normalized = np.divide(b3norm, (b1norm+b2norm+b3norm),
                        out=np.zeros_like(b3norm),
                        where=(b1norm+b2norm+b3norm)!=0)
#     print(b1normalized)  # FOR DEBUGGING
    return b1normalized, b2normalized, b3normalized


def getminmax(rbands,gbands,bbands):
    '''
    DEPRECIATED.
    Get the minimum and maximum values for each band.

    :param numpy.array b1: Input band 1
    :param numpy.array b2: Input band 2
    :param numpy.array b3: Input band 3
    :param int depth: Bit-depth of the input data
        (default is 16-bit data, which is limited to 0,65535)

    Returns three normalized bands:

        * b1normalized (:py:class:`float`)
        * b2normalized (:py:class:`float`)
        * b3normalized (:py:class:`float`)
    '''
    rmn,rmx = 99999,-99999
    gmn,gmx = 99999,-99999
    bmn,bmx = 99999,-99999
    for rb in rbands:
        if np.amin(rb)<rmn: rmn=np.amin(rb)
        if np.amax(rb)>rmx: rmx=np.amax(rb)
    for gb in gbands:
        if np.amin(gb)<gmn: gmn=np.amin(gb)
        if np.amax(gb)>gmx: gmx=np.amax(gb)
    for bb in bbands:
        if np.amin(bb)<bmn: bmn=np.amin(bb)
        if np.amax(bb)>bmx: bmx=np.amax(bb)
    return rmn, rmx, gmn, gmx, bmn, bmx

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
 '''


def otsu_getthresh(gray, rmin, rmax, nbins):
    '''
    PURPOSE:
        Function to read in an array (combined from two training class histograms) and find a threshold value that
        maximizes the interclass variability.

    RETURN:
        Returns a single threshold value.
    '''
    num_pts = len(gray)
    mean_weigth = 1.0/num_pts
    his,bins = np.histogram(gray, np.arange(rmin.astype(np.float), rmax.astype(np.float), (rmax.astype(np.float)-rmin.astype(np.float))/nbins))
    final_thresh = -1
    final_value = -1
    for t in range(1,len(bins)-1):
#         print('bin {}'.format(t))
#         print('  his[:t] = {}'.format(his[:t]))
#         print('  his[t:] = {}'.format(his[t:]))
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth
#         print('  Wb = {}'.format(Wb))
#         print('  Wf = {}'.format(Wf))
        mub = np.mean(his[:t])
        muf = np.mean(his[t:])
#         print('  mub = {}'.format(muf))
#         print('  muf = {}'.format(muf))
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = bins[t]
            final_value = value
    print('Final Threshold Returned = {}'.format(final_thresh))
    return final_thresh


def otsu_appthresh(inpts, vegidxarr, veg_noveg_thresh, reclasses=[2,4]):
    '''
    PURPOSE:
        Function to apply a previously extracted threshold value to a new numpy array.

    RETURN:
        Returns a copy of the input point cloud reclassed using the provided threshold and updated class values.
        (i.e. numpy array)
    '''
    final_pts = deepcopy(inpts)
    print('Original Classes: {}'.format(final_pts))
    final_pts[vegidxarr > veg_noveg_thresh] = reclasses[1]
    final_pts[vegidxarr < veg_noveg_thresh] = reclasses[0]
    print('UPDATED Classes: {}'.format(final_pts))
    return final_pts.astype(np.int)

def threshold_otsu(inpts, minval=0.0, maxval=1.0, nbins=1000):
    '''
    PURPOSE:
        Function to read in an array (combined from two training class histograms) and find a threshold value that
        maximizes the interclass variability.

    RETURN:
        Returns a single threshold value.
    '''
    num_pts = len(inpts)
    mean_weigth = 1.0/num_pts
    his,bins = np.histogram(inpts, np.arange(rmin.astype(np.float), rmax.astype(np.float), (rmax.astype(np.float)-rmin.astype(np.float))/nbins))
    final_thresh = -1
    final_value = -1
    for t in range(1,len(bins)-1):
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth
        mub = np.mean(his[:t])
        muf = np.mean(his[t:])
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = bins[t]
            final_value = value
    print('Otsu threshold: {}'.format(final_thresh))
    print('Otsu threshold value: {}'.format(final_value))
    return final_thresh,final_value

def mergehist(vegarr, novegarr, plothist=True):
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
    df = pd.DataFrame(vegarr)  # Two normal distributions
    df2 = pd.DataFrame(novegarr)
    df_merged = pd.concat([df, df2], ignore_index=True)
    return np.array(df_merged.values)
    if plothist:
        # weights
        df_weights = np.ones_like(df.values) / len(df_merged)
        df2_weights = np.ones_like(df2.values) / len(df_merged)
        df_merged_weights = np.ones_like(df_merged.values) / len(df_merged)
        # set plot range ()
        plt_range = (df_merged.values.min(), df_merged.values.max())
        fig,ax = plt.subplots()
        ax.hist(df.values, bins=1000, weights=df_weights, color='black', histtype='step', label='vegarr', range=plt_range)
        ax.hist(df2.values, bins=1000, weights=df2_weights, color='green', histtype='step', label='novegarr', range=plt_range)
        ax.hist(df_merged.values, bins=1000, weights=df_merged_weights, color='red', histtype='step', label='Combined', range=plt_range)
        # set plot margins and display parameters
        ax.margins(0.05)
        ax.set_ylim(bottom=0)
        ax.set_xlim([np.amin(df_merged.values), np.amax(df_merged.values)])
        plt.legend(loc='upper right')


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
    outdat = np.append(outdat, np.array([newrow]), axis=0)
    return outdat


def calc_3d_sd(coords, rad=0.5):
    # build the KDTree
    tree = cKDTree(coords, leafsize=5)
    # intiialize an empty numpy array of the same length as coords
    sd = np.zeros(len(coords))
    # iterate over every point in the dense point cloud
    # (I think this is where the real slowdown is happening)
    for count,elem in enumerate(tqdm(coords)):
        # perform spatial query on point
        result = tree.query_ball_point(elem, r=rad)
        # if at least one other point is returned, then continue
        if len(result) > 0:
            # intialize 'sums' var to track sum
            sums = 0
            # compute standard deviation of X, Y, and Z separately
            # then combine these through the sqrt of the sum of squares to get 3D standard deviation
            for x in np.std(coords[result],axis=0):
                sums += x**2
            sd[count] = math.sqrt(sums)
        # otherwise, no points were found in the search radius, return zero
        else:
            sd[count] = 0
    # return the numpy array of computed standard deviation values all points in the cloud
    return sd

def calc_sd(coords, rad=0.5):
    # build the KDTree
    tree = cKDTree(coords, leafsize=5)
    # intiialize an empty numpy array of the same length as coords
    sd = np.zeros(shape=(len(coords),3), dtype=float)
    # iterate over every point in the dense point cloud
    # (I think this is where the real slowdown is happening)
    for count,elem in enumerate(tqdm(coords)):
        # perform spatial query on point
        result = tree.query_ball_point(elem, r=rad)
        # if at least one other point is returned, then continue
        if len(result) > 1:
            # compute standard deviation of X, Y, and Z separately
            sd[count] = np.std(coords[result],axis=0)
        # otherwise, no points were found in the search radius, return zero
        else:
            sd[count] = 0
    # return the numpy array of computed standard deviation values all points in the cloud
    a,b,c = np.split(sd,3,axis=1)
    return a.flatten(),b.flatten(),c.flatten()

''' NOT SURE IF THIS NEXT FUNCTION WORKS '''
# def calc_3d_sd2(coords, rad=0.5):
#     # build the KDTree
#     tree = cKDTree(coords, leafsize=5)
#     # intiialize an empty numpy array of the same length as coords
#     sd = np.zeros((len(coords), 3))
#     # iterate over every point in the dense point cloud
#     # (I think this is where the real slowdown is happening)
#     # perform spatial query on point
#     results = tree.query_ball_point(coords, r=rad)
#     for count, result in enumerate(results):
#         # if at least one other point is returned, then continue
#         if len(result) > 1:
#             # compute standard deviation
#             sd[count] = np.std(coords[result], axis=0)
#         # otherwise, no points were found in the search radius, return zero
#         else:
#             sd[count] = 0
#     # return the numpy array of computed standard deviation values all points in the cloud
#     return np.sqrt(np.sum(sd ** 2, axis=1))


'''
The following are a collection of functions to compute statistics
for the computed vegetation indices.
'''
def addstats(intable, values_column='vals'):
    minar,maxar,medar,meanar,stdar = [],[],[],[],[]
    for index,row in intable.iterrows():
        minar.append(np.amin(row[values_column]))
        maxar.append(np.amax(row[values_column]))
        medar.append(np.median(row[values_column]))
        meanar.append(np.mean(row[values_column]))
        stdar.append(np.std(row[values_column]))
    intable['min'] = minar
    intable['max'] = maxar
    intable['med'] = medar
    intable['mean'] = meanar
    intable['std'] = stdar
    del([minar,maxar,medar,meanar,stdar])

def computeM_singletable(intable):
    '''
    Compute the M-statistic to compare all rows in the input table.
    From:
        Kaufman, Y.J.; Remer, L.A. Detection of forests using mid-IR reflectance: An application for aerosol studies.
        IEEE Trans. Geosci. Remote Sens. 1994, 32, 672–683.
    '''
    outtable = pd.DataFrame(0.00, index=list(intable.index),
                            columns=list(intable.index))
    for i in list(intable.index):
        for j in list(intable.index):
            if (i!=j)&(intable['mean'][i]!=intable['mean'][j]):
                outtable[i][j] = (intable['mean'][i]-intable['mean'][j])/(intable['std'][i]-intable['std'][j])
    return outtable

def computeM(intable1, intable2, writeout=''):
    '''
    Compute the M-statistic to compare all rows in the input table.
    From:
        Kaufman, Y.J.; Remer, L.A. Detection of forests using mid-IR reflectance: An application for aerosol studies.
        IEEE Trans. Geosci. Remote Sens. 1994, 32, 672–683.
    '''
    outtable = pd.DataFrame(0.00, index=list(intable1.index), columns=['M-statistic'])
    for i in list(intable1.index):
        for j in list(intable2.index):
            if i==j:
                outtable['M-statistic'][i] = (intable1['mean'][i]-intable2['mean'][j])/(intable1['std'][i]-intable2['std'][j])
    if writeout!='':
        outtable.to_csv(str(writeout)+'_veg_noveg_M-statistic.csv')
    return outtable

# def veg_noveg(infile, vegtrain, novegtrain, idxvals, thresh):
#     # open an output file to read and write
#     outfile = file.File('__out.laz',mode='wr',header=indat.header)
#     outfile.points = indat.points  # copy the points from the specified input las/laz file
#     ptclass = deepcopy(indat.classification)  # create a copy of the input points
# def binarization(indat, inthresh):
