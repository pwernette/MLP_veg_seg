"""
@author: pwernette@usgs.gov
(copyright 2021 by Phil Wernette; pwernette@usgs.gov)

Description:
    This program will compute the following vegetation indices, their M-statistics, and Otsu threshold values:
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

    *denotes vegetation indices that appear to be unstable in simulated values (i.e. their values are not properly constrained)

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

from src.functions import *
from src.dat_norm_and_format import *
from src.veg_indices import *

# load autoreload
if '__IPYTHON__' in globals():
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext autoreload')
    ipython.magic('aimport src.functions')
    ipython.magic('aimport src.dat_norm_and_format')
    ipython.magic('aimport src.veg_indices')
    ipython.magic('autoreload 1')

# def vegidx(r,g,b, indices='all'):
#     pdindex,pdindexnames,minarr,maxarr = [],[],[],[]
#     if indices=='all' or indices=='exr' or indices=='exgr':
#         # Excess Red (ExR)
#         #    Meyer, G.E.; Neto, J.C. Verification of color vegetation indices for automated crop imaging applications.
#         #    Comput. Electron. Agric. 2008, 63, 282–293.
#         exr = 1.4*b-g
#         pdindex.append([exr])
#         pdindexnames.append('exr')
#         minarr.append(-1.0)
#         maxarr.append(1.4)
#     if indices=='all' or indices=='exg' or indices=='exgr':
#         # Excess Green (ExG)
#         #    Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color Indices forWeed Identification Under
#         #    Various Soil, Residue, and Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
#         exg = 2*g-r-b
#         pdindex.append([exg])
#         pdindexnames.append('exg')
#         minarr.append(-1.0)
#         maxarr.append(2.0)
#     # if indices=='all' or indices=='vari':
#     #     # Visible Atmospherically Resistant Index (VARI)
#     #     #    Gitelson, A.A.; Kaufman, Y.J.; Stark, R.; Rundquist, D. Novel algorithms for remote estimation of vegetation
#     #     #    fraction. Remote Sens. Environ. 2002, 80, 76–87.
#     #     vari = np.divide((g-r), (g+r-b), out=np.zeros_like(g-r), where=(g+r-b)!=0)
#     #     pdindex.append([vari])
#     #     pdindexnames.append('vari')
#     # if indices=='all' or indices=='wi':
#     #     # Woebbecke Index (WI)
#     #     #    Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color Indices forWeed Identification Under
#     #     #    Various Soil, Residue, and Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
#     #     wi = np.divide((g-b), abs(r-g), out=np.zeros_like((g-b)), where=((r-g)!=0)&((g-b)!=0))
#     #     pdindex.append([wi])
#     #     pdindexnames.append('wi')
#     if indices=='all' or indices=='exb':
#         # Excess Blue (ExB)
#         #    Mao,W.;Wang, Y.;Wang, Y. Real-time detection of between-row weeds using machine vision. In Proceedings
#         #    of the 2003 ASAE Annual Meeting; American Society of Agricultural and Biological Engineers, Las Vegas,
#         #    NV, USA, 27–30 July 2003.
#         exb = 1.4*r-g
#         pdindex.append([exb])
#         pdindexnames.append('exb')
#         minarr.append(-1.0)
#         maxarr.append(1.4)
#     if indices=='all' or indices=='exgr':
#         # Excess Green minus R (ExGR)
#         #    Neto, J.C. A combined statistical-soft computing approach for classification and mapping weed species in
#         #    minimum -tillage systems. Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln, NE, USA, August 2004.
#         exgr = exg-exr
#         pdindex.append([exgr])
#         pdindexnames.append('exgr')
#         minarr.append(-2.4)
#         maxarr.append(3.0)
#     # if indices=='all' or indices=='cive':
#     #     # calculate Color Index of Vegetation Extraction (CIVE)
#     #     cive = 0.4412*r-0.811*g+0.385*b+18.78745
#     #     pdindex.append([cive])
#     #     pdindexnames.append('cive')
#     if indices=='all' or indices=='ngrdi':
#         # calculate Normal Green-Red Difference Index (NGRDI)
#         #    Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
#         #    Environ. 1979, 8, 127–150.
#         ngrdi = np.divide((g-r), (r+g), out=np.zeros_like((g-r)), where=(g+r)!=0)
#         pdindex.append([ngrdi])
#         pdindexnames.append('ngrdi')
#         minarr.append(-1.0)
#         maxarr.append(1.0)
#     if indices=='all' or indices=='mgrvi':
#         # Modified Green Red Vegetation Index (MGRVI)
#         #    Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
#         #    Environ. 1979, 8, 127–150.
#         mgrvi = np.divide((np.power(g,2)-np.power(r,2)), (np.power(r,2)+np.power(g,2)), out=np.zeros_like((g-r)), where=(np.power(g,2)+np.power(r,2))!=0)
#         pdindex.append([mgrvi])
#         pdindexnames.append('mgrvi')
#         minarr.append(-1.0)
#         maxarr.append(1.0)
#     if indices=='all' or indices=='gli':
#         # Green Leaf Index (GLI)
#         #    Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
#         #    documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
#         gli = np.divide(2*g-r-b, 2*g+r+b, out=np.zeros_like((2*g-r-b)), where=(2*g+r+b)!=0)
#         pdindex.append([gli])
#         pdindexnames.append('gli')
#         minarr.append(-1.0)
#         maxarr.append(1.0)
#     if indices=='all' or indices=='rgbvi':
#         # Red Green Blue Vegetation Index (RGBVI)
#         #    Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.; Broscheit, J.; Gnyp, M.L.; Bareth, G. Combining
#         #    UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass
#         #    monitoring in barley. Int. J. Appl. Earth Obs. Geoinf. 2015, 39, 79–87.
#         rgbvi = np.divide((np.power(g,2)-b*r), (np.power(g,2)+b*r), out=np.zeros_like((np.power(g,2)-b*r)), where=(np.power(g,2)+b*r)!=0)
#         pdindex.append([rgbvi])
#         pdindexnames.append('rgbvi')
#         minarr.append(-1.0)
#         maxarr.append(1.0)
#     if indices=='all' or indices=='ikaw':
#         # Kawashima Index (IKAW)
#         #    Kawashima, S.; Nakatani, M. An algorithm for estimating chlorophyll content in leaves using a video camera.
#         #    Ann. Bot. 1998, 81, 49–54.
#         ikaw = np.divide((r-b), (r+b), out=np.zeros_like((r-b)), where=(r+b)!=0)
#         pdindex.append([ikaw])
#         pdindexnames.append('ikaw')
#         minarr.append(-1.0)
#         maxarr.append(1.0)
#     if indices=='all' or indices=='gla':
#         # Green Leaf Algorithm (GLA)
#         #    Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
#         #    documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
#         gla = np.divide((2*g-r-b), (2*g+r+b), out=np.zeros_like((2*g-r-b)), where=(2*g+r+b)!=0)
#         pdindex.append([gla])
#         pdindexnames.append('gla')
#         minarr.append(-1.0)
#         maxarr.append(1.0)
#     # if indices=='all' or indices=='vega':
#     #     # Vegetativen (vega)
#     #     #    Hague, T.; Tillett, N.D.; Wheeler, H. Automated crop and weed monitoring in widely spaced cereals. Precis.
#     #     #    Agric. 2006, 7, 21–32.
#     #     vega = np.divide(g, (np.power(r,0.667)*np.power(b,(1-0.667))), out=np.zeros_like(g), where=(np.power(r,0.667)*np.power(b,(1-0.667)))!=0)
#     #     pdindex.append([vega])
#     #     pdindexnames.append('vega')
#     # if indices=='all' or indices=='com':
#     #     # calculate a combined (COM) vegetation index from:
#     #     #    Yang, W., Wang, S., Zhao, X., Zhang, J., and Feng, J. (2015). Greenness identification based on
#     #     #    HSV decision tree. Information Processing in Agriculture 2, 149–160. doi:10.1016/j.inpa.2015.07.003.
#     #     com = 0.25*exg+0.30*exgr+0.33*cive+0.12*vega
#     #     pdindex.append([com])
#     #     pdindexnames.append('com')
#     # Use vegetation indices names as the index (row labels) for the dataframe
#     # FIRST, create the data frame with the raw index values
#     outdat = pd.DataFrame(pdindex,
#                           index=pdindexnames,
#                           columns=['vals'])
#     outdat['minidxpos'] = minarr   # then add a new column with the minimum possible index values
#     outdat['maxidxpos'] = maxarr   # then add a new column with the maximum possible index values
#     return outdat

def main():
    # check the laspy version
    laspyversion = int(laspy.__version__.split('.')[0])
    print('Found laspy version {}'.format(laspy.__version__))

    # get filenames
    infile_veg = getfile(window_title='Select Vegetation Training Sample')
    infile_noveg = getfile(window_title='Select Non-Vegetation Training Sample')

    # start timer
    startTime = time.time()

    # Read in the complete LAS/LAZ file and the vegetation/no-vegetation LAS/LAZ Files
    if laspyversion == 1:
        try:
            print('Reading: '+str(infile_veg))
            fnoveg = file.File(str(infile_veg),mode='r')
            print('  No Vegetation Point Cloud: {}'.format(len(fnoveg)))
            print('Reading: '+str(infile_noveg))
            fveg = file.File(str(infile_noveg),mode='r')
            print('  Vegetation ONLY Point Cloud: {}'.format(len(fveg)))
        except Exception as e:
            sys.exit(e)
    elif laspyversion == 2:
        try:
            print('Reading: '+str(infile_veg))
            fnoveg = laspy.read(str(infile_veg))
            print('  No Vegetation Point Cloud: {}'.format(len(fnoveg)))
            print('Reading: '+str(infile_noveg))
            fveg = laspy.read(str(infile_noveg))
            print('  Vegetation ONLY Point Cloud: {}'.format(len(fveg)))
        except Exception as e:
            sys.exit(e)

    # extract the base filename from the input sample LAS/LAZ file header
    root_dir,infilename1 = os.path.split(infile_veg)
    infilename1 = infilename1.split('.')[0]
    root_dir,infilename2 = os.path.split(infile_veg)
    infilename2 = infilename2.split('.')[0]

    '''
    Compute vegetation index/indices

    This function call first performs band normalization and is robust to 8- and
    16-bit color depths (automatically extracted based on the R,G,B values).
    After band normalization, all vegetation indices are computed for each point
    in the input training dense point clouds.
    '''
    # compute vegetation indices
    print('Computing vegetation indices...')
    veg_idx = vegidx(fveg, indices=['all'])
    noveg_idx = vegidx(fnoveg, indices=['all'])

    # clean up workspace (if laspy major version == 1.x.x)
    if laspyversion == 1:
        try:
            fveg.close()
            fnoveg.close()
        except Exception as e:
            print(e)
            pass

    # compute the summary stats of each index and append to the index table
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
