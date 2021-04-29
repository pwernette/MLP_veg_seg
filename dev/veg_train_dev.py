"""
@author: pwernette@usgs.gov
(copyright 2021 by Phil Wernette; pwernette@usgs.gov)

Description:
    This program will compute the following vegetation indicies, their M-statistics, and Otsu threshold values:
        Excess Red (exr)
        Excess Green (exg)
        Excess Green-Red (exgr)
        Excess Blue (exb)
        Normal Green-Red Difference Index (ngrdi)
        Modified Green Red Vegetation Index (mgrvi)
        Green Leaf Index (gli)
        Red Green Blue Veggetation Index (rgbvi)
        Kawashima Index (ikaw)
        Green Lead Algorithm (gla)
        Visible Atmospherically Resistant Index (vari)*
        Woebbecke Index (wi)*
        Color Index of Vegetation Extraction (cive)*
        Vegetation (vega)*
        Combined Vegetation Index (com)*
            --> This index is a combination of (1) exg, (2) exgr, (3) cive, and (4) vega

    *denotes vegetation indicies that appear to be unstable in simulated values (i.e. their values are not properly constrained)

Usage:
    python veg_train.py

Inputs:
    The program will automatically request the user to select 2 input files:
        1) The point cloud containing vegetation points only (for training).
        2) The point cloud containing only bare-Earth points (for training).

Outputs:
    An output CSV file will be generated with the following naming scheme:
        {veg_filename}_{noveg_filename}.csv
    where {veg_filename} is the file name of the point cloud containing vegetation points only, and {noveg_filename} is the name
    of the point cloud containing bare-Earth points only.

    The output CSV will have the following attributes (columns) of information:
        {vegetation_index_name}     {minimum_possible_index_value}      {maximum_possible_index_value}      {M-statistic}       {Otsu_threshold_value}

Required Python modules:
    os
    ntpath
    time
    copy (deepcopy)
    laspy
    numpy
    pandas
    tKinter
"""
# basic libraries
import os
import ntpath
import time
from copy import deepcopy

# import laspy
import laspy
from laspy import file

# import libraries for managing data
import numpy as np

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
Function to convert an array to numpy.float32 type.
'''
def arr2float(inarrlist):
    outarr = [inarr.astype(np.float32) for inarr in inarrlist]
    inarr = outarr

'''
Normalize values in the input array with the specified min and max values to the output range normrange[].
'''
def normdat(inarr, minval=0, maxval=65535, normrange=[0,1]):
    if inarr.dtype != np.float32:
        inarr = inarr.astype(np.float32)
    if minval!=0:
        minval = np.amin(inarr)
    if maxval!=65535:
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
    if ((rmx<=65536) and (rmx>256)) or ((gmx<=65536) and (gmx>256)) or ((bmx<=65536) and (bmx>256)):
        rmx,gmx,bmx = 65536,65536,65536
    elif (rmx<=256) or (gmx<=256) or (bmx<=256):
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

'''
PURPOSE:
    Compute one or more vegetation indicies, as specified by the 'indicies' argument.

RETURN:
    Returns a pandas DataFrame object with the index (row names) as the vegetation index name and a 'values' column containing
    numpy arrays of the index values.
'''
def vegidx(r,g,b, indicies='all'):
    pdindex,pdindexnames,minarr,maxarr = [],[],[],[]
    if indicies=='all' or indicies=='exr' or indicies=='exgr':
        # Excess Red (ExR)
        #    Meyer, G.E.; Neto, J.C. Verification of color vegetation indices for automated crop imaging applications.
        #    Comput. Electron. Agric. 2008, 63, 282–293.
        exr = 1.4*b-g
        pdindex.append([exr])
        pdindexnames.append('exr')
        minarr.append(-1.0)
        maxarr.append(1.4)
    if indicies=='all' or indicies=='exg' or indicies=='exgr':
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
    # FIRST, create the data frame with the raw index values
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
            This function assumes that intable1 and intable2 have the same vegetation indicies computed and reported in the same order. If the vegidx()
            function was used to compute the vegetation indicies then this assumption should be valid.
        '''
        outdat['M-statistic'][idx] = (indat1['mean'][idx]-indat2['mean'][idx])/(indat1['std'][idx]-indat2['std'][idx])
        # Combine vegetation index values from vegetation and non-vegetation classes
        comboidx = mergehist(indat1['vals'][idx], indat2['vals'][idx])
        # Compute the appropriate binarization threshold value for veg/no-veg samples using Otsu's thresholding method
        # function otsu_getthresh().
        outdat['threshold'][idx] = otsu_getthresh(comboidx, outdat['minpossiblevalue'][idx], outdat['maxpossiblevalue'][idx], nbins=1000)
    return outdat

def main():
    # get filenames
    infile_veg = getfile(window_title='Select Vegetation Training Sample')
    infile_noveg = getfile(window_title='Select Non-Vegetation Training Sample')
    # infile_veg = 'C:/Users/pwernette/OneDrive - DOI/Machine Learning/test_dat/training_veg.laz'
    # infile_noveg = 'C:/Users/pwernette/OneDrive - DOI/Machine Learning/test_dat/training_noveg.laz'

    # get start time
    startTime = time.time()

    # Read in the complete LAS/LAZ file and the vegetation/no-vegetation LAS/LAZ Files
    print('Reading: '+str(infile_veg))
    fnoveg = file.File(str(infile_veg),mode='r')
    print('  No Vegetation Point Cloud: {}'.format(len(fnoveg)))
    print('Reading: '+str(infile_noveg))
    fveg = file.File(str(infile_noveg),mode='r')
    print('  Vegetation ONLY Point Cloud: {}'.format(len(fveg)))

    # extract the base filename from the input sample LAS/LAZ file header
    root_dir,infilename1 = os.path.split(fveg.filename)
    infilename1 = infilename1.split('.')[0]
    root_dir,infilename2 = os.path.split(fnoveg.filename)
    infilename2 = infilename2.split('.')[0]

    '''
    NOTE: Use deepcopy whenever copying variables, as it creates a stand-alone version of the variable and values,
    instead of simply referencing back to the original variable and values. Without deepcopy(), modifying a
    "copied object" may actually result in modifying the original object, instead of the copy.

    NEED a deepcopy() of each band for normalizing each band (see next code block)
    '''
    print('Creating copies of each input file r, g, and b values...')
    # rv,gv,bv = deepcopy(fveg.red),deepcopy(fveg.green),deepcopy(fveg.blue)  # vegetation pc
    rvcopy,gvcopy,bvcopy = deepcopy(fveg.red),deepcopy(fveg.green),deepcopy(fveg.blue)

    # rn,gn,bn = deepcopy(fnoveg.red),deepcopy(fnoveg.green),deepcopy(fnoveg.blue)  # no veg. pc
    rncopy,gncopy,bncopy = deepcopy(fnoveg.red),deepcopy(fnoveg.green),deepcopy(fnoveg.blue)

    # clean up workspace
    fveg.close()
    fnoveg.close()

    '''
    Extract the minimum and maximum values of the r,g,b bands for all combined point clouds (training and testing).

    In this case, since we are dealing with point clouds that have 16-bit color depth, the minimum value possible
    is 0 and the maximum value possible is 65,535. If changing from 16-bit color to 8-bit color comment out the line
    with ranges 0 to 65,536 and uncomment the line with ranges 0 to 256.
    '''
    # rmin,rmax,gmin,gmax,bmin,bmax = getminmax([rcopy,rvcopy,rncopy],[gcopy,gvcopy,gncopy],[bcopy,bvcopy,bncopy])
    rmin,rmax,gmin,gmax,bmin,bmax = getminmax([rvcopy,rncopy],[gvcopy,gncopy],[bvcopy,bncopy])

    print('Normalizing R, G, B bands...')
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
    print('Computing vegetation indicies...')
    veg_idx = vegidx(rv,gv,bv)
    noveg_idx = vegidx(rn,gn,bn)

    # compute the summary stats (minimum, maximum, median, mean, and standard deviations) of each index and append to the index table
    addstats(veg_idx)
    addstats(noveg_idx)

    '''
    Compute the appropriate binarization threshold value for veg/no-veg samples using Otsu's thresholding method
    function otsu().
    '''
    print('Computing threshold values for each vegetation index...')
    outtable = otsu(veg_idx, noveg_idx)
    del(veg_idx,noveg_idx)
    # write out the index thresholds
    outtable.to_csv(ntpath.join(root_dir,(infilename1+'_'+infilename2+'.csv')).replace('\\','/'))
    print('Index names and threshold values written to {}'.format(str(ntpath.join(root_dir,(infilename1+'_'+infilename2+'.csv')).replace('\\','/'))))

if __name__ == '__main__':
    main()
