"""
@author: pwernette@usgs.gov
(copyright 2021 by Phil Wernette; pwernette@usgs.gov)

Description:
    This program will reclassify a LAS/LAZ file point cloud based on a vegetation index selected from the following list:
        Excess Red (exr)
        Excess Green (exg)
        Excess Green-Red (exgr)
        Excess Blue (exb)
        Normal Green-Red Difference Index (ngrdi)
        Modified Green Red Vegetation Index (mgrvi)
        Green Leaf Index (gli)
        Red Green Blue Veggetation Index (rgbvi)
        Kawashima Index (ikaw)
        Green Leaf Algorithm (gla)
        Visible Atmospherically Resistant Index (vari)*
        Woebbecke Index (wi)*
        Color Index of Vegetation Extraction (cive)*
        Vegetation (vega)*
        Combined Vegetation Index (com)*
            --> This index is a combination of (1) exg, (2) exgr, (3) cive, and (4) vega

    *denotes vegetation indicies that appear to be unstable in simulated values (i.e. their values are not properly constrained)

Usage:
    python vegfilter.py

Inputs:
    The program will automatically request the user to select 3 input files:
        1) The point cloud to be reclassified.
        2) The point cloud containing vegetation points only (for training).
            This input point cloud contains only points representative of vegetation and can be segmented out of the larger point
            cloud using the Segment tool in CloudCompare. Once the point cloud has been segmented, export the resuling sample
            as a LAS or LAZ file.
        3) The point cloud containing only bare-Earth points (for training).
            This input point cloud contains only points representative of bare-Earth and can be segmented out of the larger point
            cloud using the Segment tool in CloudCompare. Once the point cloud has been segmented, export the resuling sample
            as a LAS or LAZ file.

Outputs:
    A new LAZ file will be generated with the following naming scheme:
        {filename}_reclass_{maxidx}_veg_noveg.laz
    where {filename} is the original point cloud file name and {maxidx} is the name of the vegetation index selected as the most
    appropriate to differentiate vegetation from bare-Earth using Otsu's thresholding approach. The output LAZ file will be saved
    in the same directory as the input file and will contain all the original points with updated classification values
    corresponding to vegetation or bare-Earth.

Required Python modules:
    os
    ntpath
    time
    copy (deepcopy)
    subprocess
    numpy
    matplotlib
    pandas
    laspy
    tKinter
"""
class Args():
    """ Simple class to hold arguments """
    pass
defaults = Args()

# OPTIONAL INPUTS
# If you wanted to manually specify which vegetation index to use, then set defaults.veg_index equal to that abbreviation
defaults.veg_index = ''
defaults.thresholding_bins = 10000  # the number of bins is purely for vegetation thresholding

# basic libraries
import os
import ntpath
import time
from copy import deepcopy
from subprocess import call

# import laspy
import laspy
from laspy import file

# import libraries for managing and plotting data
import numpy as np
import matplotlib.pyplot as plt

# import data frame libraries
import pandas as pd

# message box and file selection libraries
import tkinter
from tkinter import Tk
from tkinter.filedialog import askopenfile
from tkinter.filedialog import askdirectory

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
Function to open a dialog window where the user can select a directory.
'''
def getdir(window_title='Select Directory'):
    root_win = Tk()  # initialize the tKinter window to get user input
    root_win.withdraw()
    root_win.update()
    dir_io = askdirectory(title=window_title)  # get user input for the output directory
    root_win.destroy()  # destroy the tKinter window
    if not os.name=='nt':  # if the OS is not Windows, then correct slashes in directory name
        dir_io = ntpath.normpath(dir_io)
    return dir_io

'''
Function to get all LAS and LAZ files within a user-specified directory.
'''
def getpointclouds():
    # request a user-specified input directory
    in_dir = getdir(window_title='Select Directory with Input Files')
    # initialize a list of acceptable input point cloud formats (based on CloudCompare acceptable inputs)
    dc_formats = ["LAZ","laz","LAS","las"]
    # initialize empty list of dense clouds in the directory
    dc_in_dir = []
    # iterate through files in the input directory
    for file in os.listdir(in_dir):
        # exclude directories by checking that the "file" is not actually a directory
        if not os.path.isdir(ntpath.join(in_dir,file)):
            # if the file extension is in the list of acceptable point cloud formats (see above), then continue
            if file.split(".")[1] in dc_formats:
                # NOTE: since CloudCompare requires path slashes to be standardized, it is important that all forward or backslashes be OS standardized
                #       For Windows systems, all "\" and "\\" must be converted to "/"
                if os.name=='nt':  # check if the operating system is Windows
                    try:
                        dc_in_dir.append(os.path.join(in_dir,file).replace("\\", "/"))
                    except:
                        # MAY NOT WORK (depending on how the command line is configured in Windows)
                        dc_in_dir.append(os.path.join(in_dir,file))
                else:  # if NOT Windows, then the slashes should be properly configured with ntpath.join()
                    dc_in_dir.append(ntpath.join(in_dir,file))
    return dc_in_dir

'''
Function to convert an array to numpy.float32 type.
'''
def arr2float(inarrlist):
    outarr = [inarr.astype(np.float32) for inarr in inarrlist]
    inarr = outarr
    # for inarr in inarrlist:
    #     if inarr.dtype != np.float32:
    #         inarr = inarr.astype(np.float32)

'''
Normalize values in the input array with the specified min and max values to the output range normrange[].
'''
def normdat(inarr, minval=0, maxval=65536, normrange=[0,1]):
    if inarr.dtype != np.float32:
        inarr = inarr.astype(np.float32)
    if minval!=0:
        minval = np.amin(inarr)
    if maxval!=65536:
        maxval = np.amax(inarr)
    norminarr = (normrange[1]-normrange[0])*np.divide((inarr-minval), (maxval-minval), out=np.zeros_like(inarr), where=(maxval-minval)!=0)-normrange[0]
    return np.asarray(norminarr)

'''
Normalize each band by R+G+B.
'''
def normRGB(inarr, Rband, Gband, Bband):
    arr2float([inarr,Rband,Gband,Bband])
    normRGB = np.divide(inarr, (Rband+Gband+Bband), out=np.zeros_like(inarr), where=(Rband+Gband+Bband)!=0)
    return np.asarray(normRGB)

'''
Get the minimum and maximum values for each band.
'''
def getminmax(rbands,gbands,bbands):
    rmn,rmx = 0,-99999
    gmn,gmx = 0,-99999
    bmn,bmx = 0,-99999
    for rb in rbands:
        if np.amax(rb)>rmx: rmx=np.amax(rb)
    for gb in gbands:
        if np.amax(gb)>gmx: gmx=np.amax(gb)
    for bb in bbands:
        if np.amax(bb)>bmx: bmx=np.amax(bb)
    if ((rmx>256) and (rmx<=65536)) or ((gmx>256) and (gmx<=65536)) or ((bmx>256) and (bmx<=65536)):
        rmx,gmx,bmx = 65536,65536,65536
    elif ((rmx>0) and (rmx<=256)) or ((gmx>0) and (gmx<=256)) or ((bmx>0) and (bmx<=256)):
        rmx,gmx,bmx = 256,256,256
    return rmn, rmx, gmn, gmx, bmn, bmx

'''
Otsu's thresholding function

Function is adapted from the following reference:
    Otsu, N. A threshold selection method from gray level histogram. IEEE Trans. Syst. Man Cybern. 1979, 9, 66–166.

The source for the code block is:
    https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python
     NOTE: The original answer does not include intensity. However, the follow-up answer (see the second
     code block) DOES include intensity in the computation of a threshold.
     Since we are trying to adapt this approach to only work with vegetation indicies, we do not need to
     include the recorded intensity when thresholding.
 '''
# def otsu(gray):  # DOES NOT INCLUDE INTENSITY
#     pixel_number = gray.shape[0] * gray.shape[1]
#     mean_weigth = 1.0/pixel_number
#     his, bins = np.histogram(gray, np.array(range(0, 256)))  # assumes 8-bit color depth
#     his, bins = np.histogram(gray, np.array(range(0, 65536)))  # assumes 16-bit color depth
#     final_thresh = -1
#     final_value = -1
#     for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
#         Wb = np.sum(his[:t]) * mean_weigth
#         Wf = np.sum(his[t:]) * mean_weigth
#         # compute the mean of each distribution
#         mub = np.mean(his[:t])
#         muf = np.mean(his[t:])
#         # compute the threshold value
#         value = Wb * Wf * (mub - muf) ** 2
#         # for debugging
# #         print("Wb", Wb, "Wf", Wf)
# #         print("t", t, "value", value)
#         # if the new value is greater than the starting threshold value, update the final_value
#         if value > final_value:
#             final_thresh = t
#             final_value = value
#     final_img = gray.copy()
#     print(final_thresh)
#     final_img[gray > final_thresh] = 255
#     final_img[gray < final_thresh] = 0
#     return final_img
# def otsu(gray):  # INCLUDES INTENSITY
#     pixel_number = gray.shape[0] * gray.shape[1]
#     mean_weight = 1.0/pixel_number
#     his, bins = np.histogram(gray, np.arange(0,256))  # assumes 8-bit color depth
# #     his, bins = np.histogram(gray, np.arange(0,65536))  # assumes 16-bit color depth
#     final_thresh = -1
#     final_value = -1
#     intensity_arr = np.arange(256)
#     for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
#         pcb = np.sum(his[:t])
#         pcf = np.sum(his[t:])
#         Wb = pcb * mean_weight
#         Wf = pcf * mean_weight
#         # compute the mean of the intensity multiplied by the histogram values
#         mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
#         muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
# #         print mub, muf  # for debugging
#         value = Wb * Wf * (mub - muf) ** 2
#         # if the new value is greater than the starting threshold value, update the final_value
#         if value > final_value:
#             final_thresh = t
#             final_value = value
#     final_img = gray.copy()
#     print(final_thresh)
#     final_img[gray > final_thresh] = 255
#     final_img[gray < final_thresh] = 0
#     return final_img

'''
PURPOSE:
    Function to read in an array (combined from two training class histograms) and find a threshold value that
    maximizes the interclass variability.

RETURN:
    Returns a single threshold value.
'''
def otsu_getthresh(gray, rmin, rmax, nbins):
    num_pts = len(gray)
    mean_weigth = 1.0/num_pts
    his,bins = np.histogram(gray, np.arange(rmin.astype(np.float32), rmax.astype(np.float32), (rmax.astype(np.float32)-rmin.astype(np.float32))/nbins))
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
    print('Final Threshold Returned = {}'.format(final_thresh))
    return final_thresh

'''
PURPOSE:
    Function to apply a previously extracted threshold value to a new numpy array.

RETURN:
    Returns a copy of the input point cloud reclassed using the provided threshold and updated class values.
    (i.e. numpy array)
'''
def otsu_appthresh(inpts, vegidxarr, veg_noveg_thresh, reclasses=[2,4]):
    final_pts = deepcopy(inpts)
    print('Original Classes: {}'.format(final_pts))
    final_pts[vegidxarr > veg_noveg_thresh] = reclasses[1]
    final_pts[vegidxarr < veg_noveg_thresh] = reclasses[0]
    print('UPDATED Classes: {}'.format(final_pts))
    return final_pts.astype(np.int32)

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
def mergehist(vegarr, novegarr, plothist=True):
    df = pd.DataFrame(vegarr)  # Two normal distributions
    df2 = pd.DataFrame(novegarr)
    df_merged = pd.concat([df, df2], ignore_index=True)
    return np.array(df_merged.values)
    # if plothist:
    #     # weights
    #     df_weights = np.ones_like(df.values) / len(df_merged)
    #     df2_weights = np.ones_like(df2.values) / len(df_merged)
    #     df_merged_weights = np.ones_like(df_merged.values) / len(df_merged)
    #     # set plot range ()
    #     plt_range = (df_merged.values.min(), df_merged.values.max())
    #     fig,ax = plt.subplots()
    #     ax.hist(df.values, bins=1000, weights=df_weights, color='black', histtype='step', label='vegarr', range=plt_range)
    #     ax.hist(df2.values, bins=1000, weights=df2_weights, color='green', histtype='step', label='novegarr', range=plt_range)
    #     ax.hist(df_merged.values, bins=1000, weights=df_merged_weights, color='red', histtype='step', label='Combined', range=plt_range)
    #     # set plot margins and display parameters
    #     ax.margins(0.05)
    #     ax.set_ylim(bottom=0)
    #     ax.set_xlim([np.amin(df_merged.values), np.amax(df_merged.values)])
    #     plt.legend(loc='upper right')
# def mergehistnumpy(vegarr, novegarr, plothist=True):
#     df = np.array(vegarr)  # Two normal distributions
#     df2 = np.array(novegarr)
#     df_merged = np.concatenate(((df, df2)))
#     return df_merged
#     if plothist:
#         # weights
#         df_weights = np.ones_like(df.values) / len(df_merged)
#         df2_weights = np.ones_like(df2.values) / len(df_merged)
#         df_merged_weights = np.ones_like(df_merged.values) / len(df_merged)
#         # set plot range ()
#         plt_range = (np.amax(df_merged), np.amax(df_merged))
#         fig,ax = plt.subplots()
#         ax.hist(df, bins=1000, weights=df_weights, color='black', histtype='step', label='vegarr', range=plt_range)
#         ax.hist(df2, bins=1000, weights=df2_weights, color='green', histtype='step', label='novegarr', range=plt_range)
#         ax.hist(df_merged, bins=1000, weights=df_merged_weights, color='red', histtype='step', label='Combined', range=plt_range)
#         # set plot margins and display parameters
#         ax.margins(0.05)
#         ax.set_ylim(bottom=0)
#         ax.set_xlim([np.amin(df_merged), np.amax(df_merged)])
#         plt.legend(loc='upper right')

'''
PURPOSE:
    Compute one or more vegetation indicies, as specified by the 'indicies' argument.

RETURN:
    Returns a pandas DataFrame object with the index (row names) as the vegetation index name and a 'values' column containing
    numpy arrays of the index values.
'''
def vegidx(r,g,b, indicies='all'):
    pdindex,pdindexnames,minarr,maxarr = [],[],[],[]
    if indicies=='all' or indicies=='exr':
        # Excess Red (ExR)
        #    Meyer, G.E.; Neto, J.C. Verification of color vegetation indices for automated crop imaging applications.
        #    Comput. Electron. Agric. 2008, 63, 282–293.
        exr = 1.4*b-g
        pdindex.append([exr])
        pdindexnames.append('exr')
        minarr.append(-1.0)
        maxarr.append(1.4)
    if indicies=='all' or indicies=='exg':
        # Excess Green (ExG)
        #    Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color Indices forWeed Identification Under
        #    Various Soil, Residue, and Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
        exg = 2*g-r-b
        pdindex.append([exg])
        pdindexnames.append('exg')
        minarr.append(-1.0)
        maxarr.append(2.0)
    # if indicies=='all' or indicies=='vari':
    #     # Visible Atmospherically Resistant Index (VARI)
    #     #    Gitelson, A.A.; Kaufman, Y.J.; Stark, R.; Rundquist, D. Novel algorithms for remote estimation of vegetation
    #     #    fraction. Remote Sens. Environ. 2002, 80, 76–87.
    #     vari = np.divide((g-r), (g+r-b), out=np.zeros_like(g-r), where=(g+r-b)!=0)
    #     pdindex.append([vari])
    #     pdindexnames.append('vari')
    # if indicies=='all' or indicies=='wi':
    #     # Woebbecke Index (WI)
    #     #    Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color Indices forWeed Identification Under
    #     #    Various Soil, Residue, and Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
    #     wi = np.divide((g-b), abs(r-g), out=np.zeros_like((g-b)), where=((r-g)!=0)&((g-b)!=0))
    #     pdindex.append([wi])
    #     pdindexnames.append('wi')
    if indicies=='all' or indicies=='exb':
        # Excess Blue (ExB)
        #    Mao,W.;Wang, Y.;Wang, Y. Real-time detection of between-row weeds using machine vision. In Proceedings
        #    of the 2003 ASAE Annual Meeting; American Society of Agricultural and Biological Engineers, Las Vegas,
        #    NV, USA, 27–30 July 2003.
        exb = 1.4*r-g
        pdindex.append([exb])
        pdindexnames.append('exb')
        minarr.append(-1.0)
        maxarr.append(1.4)
    if indicies=='all' or indicies=='exgr':
        # Excess Green minus R (ExGR)
        #    Neto, J.C. A combined statistical-soft computing approach for classification and mapping weed species in
        #    minimum -tillage systems. Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln, NE, USA, August 2004.
        exgr = exg-exr
        pdindex.append([exgr])
        pdindexnames.append('exgr')
        minarr.append(-2.4)
        maxarr.append(3.0)
    # if indicies=='all' or indicies=='cive':
    #     # calculate Color Index of Vegetation Extraction (CIVE)
    #     cive = 0.4412*r-0.811*g+0.385*b+18.78745
    #     pdindex.append([cive])
    #     pdindexnames.append('cive')
    if indicies=='all' or indicies=='ngrdi':
        # calculate Normal Green-Red Difference Index (NGRDI)
        #    Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
        #    Environ. 1979, 8, 127–150.
        ngrdi = np.divide((g-r), (r+g), out=np.zeros_like((g-r)), where=(g+r)!=0)
        pdindex.append([ngrdi])
        pdindexnames.append('ngrdi')
        minarr.append(-1.0)
        maxarr.append(1.0)
    if indicies=='all' or indicies=='mgrvi':
        # Modified Green Red Vegetation Index (MGRVI)
        #    Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
        #    Environ. 1979, 8, 127–150.
        mgrvi = np.divide((np.power(g,2)-np.power(r,2)), (np.power(r,2)+np.power(g,2)), out=np.zeros_like((g-r)), where=(np.power(g,2)+np.power(r,2))!=0)
        pdindex.append([mgrvi])
        pdindexnames.append('mgrvi')
        minarr.append(-1.0)
        maxarr.append(1.0)
    if indicies=='all' or indicies=='gli':
        # Green Leaf Index (GLI)
        #    Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
        #    documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
        gli = np.divide(2*g-r-b, 2*g+r+b, out=np.zeros_like((2*g-r-b)), where=(2*g+r+b)!=0)
        pdindex.append([gli])
        pdindexnames.append('gli')
        minarr.append(-1.0)
        maxarr.append(1.0)
    if indicies=='all' or indicies=='rgbvi':
        # Red Green Blue Vegetation Index (RGBVI)
        #    Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.; Broscheit, J.; Gnyp, M.L.; Bareth, G. Combining
        #    UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass
        #    monitoring in barley. Int. J. Appl. Earth Obs. Geoinf. 2015, 39, 79–87.
        rgbvi = np.divide((np.power(g,2)-b*r), (np.power(g,2)+b*r), out=np.zeros_like((np.power(g,2)-b*r)), where=(np.power(g,2)+b*r)!=0)
        pdindex.append([rgbvi])
        pdindexnames.append('rgbvi')
        minarr.append(-1.0)
        maxarr.append(1.0)
    if indicies=='all' or indicies=='ikaw':
        # Kawashima Index (IKAW)
        #    Kawashima, S.; Nakatani, M. An algorithm for estimating chlorophyll content in leaves using a video camera.
        #    Ann. Bot. 1998, 81, 49–54.
        ikaw = np.divide((r-b), (r+b), out=np.zeros_like((r-b)), where=(r+b)!=0)
        pdindex.append([ikaw])
        pdindexnames.append('ikaw')
        minarr.append(-1.0)
        maxarr.append(1.0)
    if indicies=='all' or indicies=='gla':
        # Green Leaf Algorithm (GLA)
        #    Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
        #    documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
        gla = np.divide((2*g-r-b), (2*g+r+b), out=np.zeros_like((2*g-r-b)), where=(2*g+r+b)!=0)
        pdindex.append([gla])
        pdindexnames.append('gla')
        minarr.append(-1.0)
        maxarr.append(1.0)
    # if indicies=='all' or indicies=='vega':
    #     # Vegetativen (vega)
    #     #    Hague, T.; Tillett, N.D.; Wheeler, H. Automated crop and weed monitoring in widely spaced cereals. Precis.
    #     #    Agric. 2006, 7, 21–32.
    #     vega = np.divide(g, (np.power(r,0.667)*np.power(b,(1-0.667))), out=np.zeros_like(g), where=(np.power(r,0.667)*np.power(b,(1-0.667)))!=0)
    #     pdindex.append([vega])
    #     pdindexnames.append('vega')
    # if indicies=='all' or indicies=='com':
    #     # calculate a combined (COM) vegetation index from:
    #     #    Yang, W., Wang, S., Zhao, X., Zhang, J., and Feng, J. (2015). Greenness identification based on
    #     #    HSV decision tree. Information Processing in Agriculture 2, 149–160. doi:10.1016/j.inpa.2015.07.003.
    #     com = 0.25*exg+0.30*exgr+0.33*cive+0.12*vega
    #     pdindex.append([com])
    #     pdindexnames.append('com')
    # Use vegetation indicies names as the index (row labels) for the dataframe
    outdat = pd.DataFrame(pdindex,
                          index=pdindexnames,
                          columns=['vals'])
    outdat['minidxpos'] = minarr   # then add a new column with the minimum possible index values
    outdat['maxidxpos'] = maxarr   # then add a new column with the maximum possible index values
    return outdat

'''
PURPOSE:
    Compute and append summary statistics to the vegetation indicies DataFrame.

WARNING: This function modifies the specified 'intable'
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

'''
Compute the M-statistic to compare all rows in the input table.
From:
    Kaufman, Y.J.; Remer, L.A. Detection of forests using mid-IR reflectance: An application for aerosol studies.
    IEEE Trans. Geosci. Remote Sens. 1994, 32, 672–683.

The vegetation index with the greatest value, should provide the greatest ability to differentiate the vegetation and non-vegetation classes.
'''
def computeM(intable1, intable2, writeout=''):
    outtable = pd.DataFrame(0.00, index=list(intable1.index), columns=['M-statistic'])
    for i in list(intable1.index):
        outtable['M-statistic'][i] = (intable1['mean'][i]-intable2['mean'][i])/(intable1['std'][i]-intable2['std'][i])
    if writeout!='':
        outtable.to_csv(str(writeout)+'_veg_noveg_M-statistics.csv')
    return outtable

def main():
    # get files
    # classifyfile = getfile(window_title='Select File to Classify')
    # infile_veg = getfile(window_title='Select Vegetation Training Sample')
    # infile_noveg = getfile(window_title='Select Non-Vegetation Training Sample')
    classifyfile = 'E:/Machine Learning/test_dat/20180726.laz'      # for debugging on workstation
    infile_veg = 'E:/Machine Learning/test_dat/training_veg.laz'    # for debugging on workstation
    infile_noveg = 'E:/Machine Learning/test_dat/training_noveg.laz'   # for debugging on workstation

    # script timer
    startTime = time.time()

    # Read in the complete LAS/LAZ file and the vegetation/no-vegetation LAS/LAZ Files
    print('Reading: '+str(classifyfile))
    readtime = time.time()
    f = file.File(str(classifyfile),mode='r')
    print('  Point Cloud to Classify: {}'.format(len(f)))
    print('  Time to read: {}'.format((time.time()-readtime)))

    print('Reading: '+str(infile_veg))
    readtime = time.time()
    fnoveg = file.File(str(infile_veg),mode='r')
    print('  Bare-Earth Point Cloud: {}'.format(len(fnoveg)))
    print('  Time to read: {}'.format((time.time()-readtime)))

    print('Reading: '+str(infile_noveg))
    readtime = time.time()
    fveg = file.File(str(infile_noveg),mode='r')
    print('  Vegetation ONLY Point Cloud: {}'.format(len(fveg)))
    print('  Time to read: {}'.format((time.time()-readtime)))

    print('Completed reading input LAS/LAZ files in {} seconds.'.format((time.time()-startTime)))
    # extract the base filename from the input sample LAS/LAZ file header
    root_dir,infilename = os.path.split(f.filename)
    infilename = infilename.split('.')[0]

    '''
    NOTE: Use deepcopy whenever copying variables, as it creates a stand-alone version of the variable and values,
    instead of simply referencing back to the original variable and values. Without deepcopy(), modifying a
    "copied object" may actually result in modifying the original object, instead of the copy.

    NEED a deepcopy() of each band for normalizing each band (see next code block)
    '''
    print('Creating copies of each input file r, g, and b values...')
    r,g,b = deepcopy(f.red),deepcopy(f.green),deepcopy(f.blue)
    rcopy,gcopy,bcopy = deepcopy(r),deepcopy(g),deepcopy(b)

    rv,gv,bv = deepcopy(fveg.red),deepcopy(fveg.green),deepcopy(fveg.blue)  # vegetation pc
    rvcopy,gvcopy,bvcopy = deepcopy(rv),deepcopy(gv),deepcopy(bv)

    rn,gn,bn = deepcopy(fnoveg.red),deepcopy(fnoveg.green),deepcopy(fnoveg.blue)  # no veg. pc
    rncopy,gncopy,bncopy = deepcopy(rn),deepcopy(gn),deepcopy(bn)

    # clean up workspace
    fveg.close()
    fnoveg.close()

    '''
    Extract the minimum and maximum values of the r,g,b bands for all combined point clouds (training and testing).

    In this case, since we are dealing with point clouds that have 16-bit color depth, the minimum value possible
    is 0 and the maximum value possible is 65,535. If changing from 16-bit color to 8-bit color comment out the line
    with ranges 0 to 65,536 and uncomment the line with ranges 0 to 256.
    '''
    rmin,rmax,gmin,gmax,bmin,bmax = getminmax([rcopy,rvcopy,rncopy],[gcopy,gvcopy,gncopy],[bcopy,bvcopy,bncopy])
    # rmin,rmax,gmin,gmax,bmin,bmax = 0,256,0,256,0,256  # uncomment this line for 8-bit color depth
    # rmin,rmax,gmin,gmax,bmin,bmax = 0,65536,0,65536,0,65536  # uncomment this line for 16-bit color depth

    '''
    Normalize the testing r,g,b values
    '''
    r = normRGB(normdat(rcopy,rmin,rmax),normdat(rcopy,rmin,rmax),normdat(gcopy,gmin,gmax),normdat(bcopy,bmin,bmax))
    g = normRGB(normdat(gcopy,gmin,gmax),normdat(rcopy,rmin,rmax),normdat(gcopy,gmin,gmax),normdat(bcopy,bmin,bmax))
    b = normRGB(normdat(bcopy,bmin,bmax),normdat(rcopy,rmin,rmax),normdat(gcopy,gmin,gmax),normdat(bcopy,bmin,bmax))
    del(rcopy,gcopy,bcopy)  # clean up the workspace

    '''
    Normalize the vegetation training r,g,b values
    '''
    rv = normRGB(normdat(rvcopy,rmin,rmax),normdat(rvcopy,rmin,rmax),normdat(gvcopy,gmin,gmax),normdat(bvcopy,bmin,bmax))
    gv = normRGB(normdat(gvcopy,gmin,gmax),normdat(rvcopy,rmin,rmax),normdat(gvcopy,gmin,gmax),normdat(bvcopy,bmin,bmax))
    bv = normRGB(normdat(bvcopy,bmin,bmax),normdat(rvcopy,rmin,rmax),normdat(gvcopy,gmin,gmax),normdat(bvcopy,bmin,bmax))
    del(rvcopy,gvcopy,bvcopy)  # clean up the workspace

    '''
    Normalize the no-vegetation training r,g,b values
    '''
    rn = normRGB(normdat(rncopy,rmin,rmax),normdat(rncopy,rmin,rmax),normdat(gncopy,gmin,gmax),normdat(bncopy,bmin,bmax))
    gn = normRGB(normdat(gncopy,gmin,gmax),normdat(rncopy,rmin,rmax),normdat(gncopy,gmin,gmax),normdat(bncopy,bmin,bmax))
    bn = normRGB(normdat(bncopy,bmin,bmax),normdat(rncopy,rmin,rmax),normdat(gncopy,gmin,gmax),normdat(bncopy,bmin,bmax))
    del(rncopy,gncopy,bncopy)  # Clean up the workspace

    '''
    Since the r, g, and b values have been normalized all based on the same minimum and maximim values
        AND
    Since none of the vegetation indicies use minimum or maximum of any band
        THEN...
    Computing the multiple (or individual) vegetation indicies yields the same range of values, regardless
    of whether it is a training sample or testing sample (again, assuming the first assumption holds true).
    '''
    # compute vegetation indicies based on point cloud normalized and scaled rgb values
    vegidx_pts = vegidx(r,g,b)
    veg_idx = vegidx(rv,gv,bv)
    noveg_idx = vegidx(rn,gn,bn)

    # compute the summary stats (minimum, maximum, median, mean, and standard deviations) of each index and append to the index table
    addstats(vegidx_pts)
    addstats(veg_idx)
    addstats(noveg_idx)

    # print('Training table: Vegetation')
    # print(veg_idx[['vals','mean','std']])
    # print('Training table: Bare Earth')
    # print(noveg_idx[['vals','mean','std']])

    '''
    The M-statistic is an approach to identify how well normal distributions of the training classes,
    in this case vegetation and no vegetation, are distinguishable from each other. The greater the
    difference in distributions, the more differentiable they are from each other, and the greater the
    M-statistic will be.

    The 'writeout' option is used to write out the file as a CSV.

    Using the M-statistic is taken from:
        Mesas-Carrascosa, F.-J., de Castro, A. I., Torres-Sánchez, J., Triviño-Tarradas, P.,
        Jiménez-Brenes, F. M., García-Ferrer, A., et al. (2020). Classification of 3D Point
        Clouds Using Color Vegetation Indices for Precision Viticulture and Digitizing Applications.
        Remote Sensing 12, 317. doi:10.3390/rs12020317.
    '''
    # compute M-statistic for all vegetation indicies
    print('Computing M-statistics for vegetation indicies...')
    vegM = computeM(veg_idx, noveg_idx, writeout=infilename)
    print(vegM)  # print the M-statistic for each vegetation index

    '''
    Get the index of maximum M-statistic (assuming it provides the greatest opportunity to differentiate vegetation
    points from non-vegetation points).

    Then, compute the minimum and maximum of the combined index values, which will be used to merge the two training
    classes/samples.
    '''
    # extract index of max value from M-statistic table
    if defaults.veg_index=='':
        maxidx = vegM[['M-statistic']].idxmax()[0]
    else:
        maxidx = defaults.veg_index
    print("Binarization Index: {}".format(maxidx))

    # get min & max values of the vegetation index for all 3 input files
    minval = np.amin(np.concatenate((vegidx_pts['vals'][maxidx],veg_idx['vals'][maxidx],noveg_idx['vals'][maxidx])))
    maxval = np.amax(np.concatenate((vegidx_pts['vals'][maxidx],veg_idx['vals'][maxidx],noveg_idx['vals'][maxidx])))
    print('  Index min and max = ({}, {})'.format(minval,maxval))

    '''
    Combine vegetation index values from vegetation and non-vegetation classes
    '''
    comboidx = mergehist(veg_idx['vals'][maxidx], noveg_idx['vals'][maxidx])

    '''
    Compute the appropriate binarization threshold value for veg/no-veg samples using Otsu's thresholding method
    function otsu_getthresh().

    The 'nbins' value is simply used to determine how many bins will be used to divide the values into.
        More bins will INCREASE COMPUTATION TIME but will yield a MORE REFINED AND MOST APPROPRIATE binarization value.
        Fewer bins will DECREASE COMPUTATION TIME and yield a LESS REFINED binarization value.
    '''
    bin_thresh = otsu_getthresh(comboidx, minval, maxval, nbins=defaults.thresholding_bins)

    '''
    Apply the threshold from the veg/no-veg binarization to another point cloud using the otsu_appthresh() function.
    '''
    updated_classes = otsu_appthresh(deepcopy(f.classification), vegidx_pts['vals'][maxidx], bin_thresh, reclasses=[2,4])

    '''
    Create copy of entire point cloud, update the classification field, and save the LAS/LAZ file.
    '''
    # create a new file with an updated name ("_reclass_veg_noveg.laz" has been appended to the filename)
    f_out = file.File(str(ntpath.join(root_dir,(infilename+'_reclass_'+str(maxidx)+'_veg_noveg.las'))).replace('\\','/'), mode='w', header=f.header)
    # copy the points from the origina file to the new output file
    f_out.points = f.points
    # update the classification with the computed/binarized classes from the previous code block
    f_out.classification = updated_classes
    # close the output file
    f_out.close()
    # convert the LAS output file to a LAZ compressed file (to save on system storage)
    call(['las2las','-i',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(maxidx)+'_veg_noveg.las'))).replace('\\','/'),'-o',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(maxidx)+'_veg_noveg.laz'))).replace('\\','/')])
    # call(['rm',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(maxidx)+'_veg_noveg.las'))).replace('\\','/')])

    # print complete script runtime
    print('Complete Runtime: {}'.format((time.time()-startTime)))

if __name__ == '__main__':
    main()
