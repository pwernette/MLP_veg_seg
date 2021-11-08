"""
@author: pwernette@usgs.gov
(copyright 2021 by Phil Wernette; pwernette@usgs.gov)

Description:
    This program will reclassify an input LAS/LAZ file point cloud based on a vegetation index. The vegetation index is determined
    from the M-statistic values by default but can be specified manually by setting the defaults.veg_index to one of the following values in parentheses:
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
    python veg_reclass.py

Inputs:
    The program will automatically request the user to select 3 input files:
        1) The point cloud to be reclassified.
        2) The CSV file containing the vegetation index value ranges, M-statistics, and Otsu threshold values.
            --> This file is automatically created when you run veg_train.py

Outputs:
    A new LAZ file will be generated with the following naming scheme:
        {filename}_reclass_{vegetation_index_name}_veg_noveg.las
    where {filename} is the original point cloud file name and {vegetation_index_name} is the name of the vegetation index
    determined or selected to differentiate vegetation from bare-Earth using Otsu's thresholding approach. The output LAZ
    file will be saved in the same directory as the input file and will contain all the original points with updated classification
    values corresponding to either vegetation or bare-Earth.

Required Python modules:
    os
    ntpath
    time
    copy (deepcopy)
    subprocess (Popen)
    laspy (file)
    numpy
    pandas
    tKinter
"""
class Args():
    """ Simple class to hold arguments """
    pass
defaults = Args()

defaults.veg_index = ''

# basic libraries
import os
import ntpath
import time
from copy import deepcopy
from subprocess import Popen

# import laspy
import laspy
from laspy import file

# import libraries for managing and plotting data
import numpy as np

# import data frame libraries
import pandas as pd

# message box and file selection libraries
import tkinter
from tkinter import Tk
from tkinter.filedialog import askopenfile

from src.functions import *
from src.veg_indices import *

# load autoreload
%load_ext autoreload

# autreload selected modules
%aimport src.functions
%aimport src.veg_indices
%autoreload 1

def main():
    # check the laspy version
    laspyversion = int(laspy.__version__.split('.')[0])
    print('Found laspy version {}'.format(laspy.__version__))

    # get filenames
    classifyfile = getfile(window_title='Select Point Cloud to Reclassify')

    # get filename
    trained_csv_file = getfile(window_title='Select CSV File Created From veg_train.py')

    # get start time
    startTime = time.time()

    '''
    Read in the complete LAS/LAZ file
    '''
    if laspyversion == 1:
        try:
            print('Reading: '+str(classifyfile))
            readtime = time.time()
            f = file.File(str(classifyfile),mode='r')
            print('  Point Cloud to Classify: {}'.format(len(f)))
            print('  Time to read: {}'.format((time.time()-readtime)))
        except Exception as e:
            sys.exit(e)
    elif laspyversion == 2:
        try:
            print('Reading: '+str(classifyfile))
            readtime = time.time()
            f = laspy.read(str(classifyfile))
            print('  Point Cloud to Classify: {}'.format(len(f)))
            print('  Time to read: {}'.format((time.time()-readtime)))
        except Exception as e:
            sys.exit(e)

    '''
    Read in training table
    '''
    print('Read in trained info file: {}'.format(trained_csv_file))
    # use index_col=0: since output file uses vegetation index as the row name
    veg_otsu_infile = pd.read_csv(trained_csv_file, index_col=0)

    # extract the base filename from the input sample LAS/LAZ file header
    root_dir,infilename = os.path.split(f.filename)
    infilename = infilename.split('.')[0]

    '''
    Normalize the r,g,b values
    '''
    print('Normalizing R, G, B bands...')
    r,g,b = normBands(deepcopy(f.red), deepcopy(f.green), deepcopy(f.blue))

    '''
    Select vegetation index
    '''
    if defaults.veg_index=='':
        vegetation_index = veg_otsu_infile[['M-statistic']].idxmax()[0]
        print('{} vegetation index automatically selected based on M-statistic in input table.'.format(vegetation_index))
    else:
        vegetation_index = defaults.veg_index
        print('Using specified vegetation index: {}'.format(vegetation_index))

    '''
    Apply threshold from veg/no-veg binarization using otsu_appthresh() function
    '''
    updated_classes = apply_otsu(r,g,b, deepcopy(f.classification), vegetation_index, veg_otsu_infile['threshold'][vegetation_index], reclasses=[2,4])

    '''
    Copy point cloud, update classification field, & save new LAZ file.
    '''
    # create a new file with an updated name
    #   "_reclass_{vegetation_index_name}_veg_noveg.laz" is appended to filename
    if laspyversion == 1:
        try:
            f_out = file.File(str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/'), mode='w', header=f.header)
            # copy the points from the origina file to the new output file
            f_out.points = f.points
            # update the classification with the computed/binarized classes
            f_out.classification = updated_classes
            # close the output file
            f_out.close()
            f.close()
            del(r,g,b,updated_classes)

            # NOTE: Because laspy version 1.x.x cannot write LAZ files, Popen
            # utility is used to convert the output LAS file to a compressed
            # LAZ file (to save on storage space)
            proc = Popen(['las2las64','-i',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/'),'-o',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.laz'))).replace('\\','/')])
            proc.wait()
            (stdout,stderr) = proc.communicate()
            if proc.returncode!=0:
                print(stderr)
                proc2 = Popen(['las2las','-i',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/'),'-o',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.laz'))).replace('\\','/')])
                proc2.wait()
                (stdout2,stderr2) = proc2.communicate()
                if proc2.returncode!=0:
                    print("Unable to locate LAStools las2las utility. Please check that LAStools is installed properly.")
                    print(stderr2)
                else:
                    print('Successfully converted LAS to LAZ file using 32-bit las2las.')
                    # Remove the original output LAS file.
                    rmfile = Popen(['rm',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/')])
                    rmfile.wait()
                    (rmout,rmerr) = rmfile.communicate()
                    if rmfile.returncode!=0:
                        print(rmerr)
                    else:
                        print('Successfully deleted old LAS file to save storage space.')
            else:
                print('Successfully converted LAS to LAZ file using 64-bit las2las.')
                # Remove the original output LAS file.
                rmfile = Popen(['rm',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/')])
                rmfile.wait()
                (rmout,rmerr) = rmfile.communicate()
                if rmfile.returncode!=0:
                    print(rmerr)
                else:
                    print('Successfully deleted old LAS file to save storage space.')
        except Exception as e:
            sys.exit(e)
    elif laspyversion == 2:
        try:
            f_out = laspy.LasData(f.header)
            # copy the points from the origina file to the new output file
            f_out.points = f.points
            # update the classification with the computed/binarized classes
            f_out.classification = updated_classes
            # close the output file
            f_out.write(str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.laz'))).replace('\\','/'))
            del(r,g,b,updated_classes)
        except Exception as e:
            sys.exit(e)

    # print complete script runtime
    print('COMPLETE RUNTIME: {}'.format((time.time()-startTime)))

if __name__ == '__main__':
    main()
