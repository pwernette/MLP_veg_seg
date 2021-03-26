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

defaults.veg_index = 'rgbvi'

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
def normdat(inarr, minval, maxval, normrange=[0,1]):
    if inarr.dtype != np.float32:
        inarr = inarr.astype(np.float32)
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
def getminmax(rband,gband,bband):
    rmn,gmn,bmn = 0,0,0
    rmx = np.amax(rband)
    gmx = np.amax(gband)
    bmx = np.amax(bband)
    if ((rmx<=65536) and (rmx>256)) or ((gmx<=65536) and (gmx>256)) or ((bmx<=65536) and (bmx>256)):
        rmx,gmx,bmx = 65536,65536,65536
    elif (rmx<=256) or (gmx<=256) or (bmx<=256):
        rmx,gmx,bmx = 256,256,256
    return rmn, rmx, gmn, gmx, bmn, bmx

'''
PURPOSE:
    Compute one or more vegetation indicies, as specified by the 'indicies' argument.

RETURN:
    Returns a pandas DataFrame object with the index (row names) as the vegetation index name and a 'values' column containing
    numpy arrays of the index values.
'''
def vegidx(r,g,b, indicies='all'):
    pdindex,pdindexnames = [],[]
    if indicies=='all' or indicies=='exr'or indicies=='exgr':
        # Excess Red (ExR)
        #    Meyer, G.E.; Neto, J.C. Verification of color vegetation indices for automated crop imaging applications.
        #    Comput. Electron. Agric. 2008, 63, 282–293.
        exr = 1.4*b-g
        pdindex.append([exr])
        pdindexnames.append('exr')
    if indicies=='all' or indicies=='exg' or indicies=='exgr':
        # Excess Green (ExG)
        #    Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color Indices forWeed Identification Under
        #    Various Soil, Residue, and Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
        exg = 2*g-r-b
        pdindex.append([exg])
        pdindexnames.append('exg')
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
    if indicies=='all' or indicies=='exgr':
        # Excess Green minus R (ExGR)
        #    Neto, J.C. A combined statistical-soft computing approach for classification and mapping weed species in
        #    minimum -tillage systems. Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln, NE, USA, August 2004.
        exgr = exg-exr
        pdindex.append([exgr])
        pdindexnames.append('exgr')
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
    if indicies=='all' or indicies=='mgrvi':
        # Modified Green Red Vegetation Index (MGRVI)
        #    Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
        #    Environ. 1979, 8, 127–150.
        mgrvi = np.divide((np.power(g,2)-np.power(r,2)), (np.power(r,2)+np.power(g,2)), out=np.zeros_like((g-r)), where=(np.power(g,2)+np.power(r,2))!=0)
        pdindex.append([mgrvi])
        pdindexnames.append('mgrvi')
    if indicies=='all' or indicies=='gli':
        # Green Leaf Index (GLI)
        #    Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
        #    documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
        gli = np.divide(2*g-r-b, 2*g+r+b, out=np.zeros_like((2*g-r-b)), where=(2*g+r+b)!=0)
        pdindex.append([gli])
        pdindexnames.append('gli')
    if indicies=='all' or indicies=='rgbvi':
        # Red Green Blue Vegetation Index (RGBVI)
        #    Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.; Broscheit, J.; Gnyp, M.L.; Bareth, G. Combining
        #    UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass
        #    monitoring in barley. Int. J. Appl. Earth Obs. Geoinf. 2015, 39, 79–87.
        rgbvi = np.divide((np.power(g,2)-b*r), (np.power(g,2)+b*r), out=np.zeros_like((np.power(g,2)-b*r)), where=(np.power(g,2)+b*r)!=0)
        pdindex.append([rgbvi])
        pdindexnames.append('rgbvi')
    if indicies=='all' or indicies=='ikaw':
        # Kawashima Index (IKAW)
        #    Kawashima, S.; Nakatani, M. An algorithm for estimating chlorophyll content in leaves using a video camera.
        #    Ann. Bot. 1998, 81, 49–54.
        ikaw = np.divide((r-b), (r+b), out=np.zeros_like((r-b)), where=(r+b)!=0)
        pdindex.append([ikaw])
        pdindexnames.append('ikaw')
    if indicies=='all' or indicies=='gla':
        # Green Leaf Algorithm (GLA)
        #    Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
        #    documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
        gla = np.divide((2*g-r-b), (2*g+r+b), out=np.zeros_like((2*g-r-b)), where=(2*g+r+b)!=0)
        pdindex.append([gla])
        pdindexnames.append('gla')
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
    return pd.DataFrame(pdindex,
                          index=pdindexnames,
                          columns=['vals'])

'''
PURPOSE:
    Compute the appropriate vegetation index and reclassify a point cloud using previously extracted thresholds.
'''
def apply_otsu(inr,ing,inb, inclassvals, veg_index, otsu_threshold, reclasses=[2,4]):
    # compute vegetation indicies based on point cloud normalized and scaled rgb values
    print('Computing vegetation indicies...')
    idxvals = vegidx(inr,ing,inb, indicies=veg_index)
    # create a deep copy of the original point classification values
    final_pts = deepcopy(inclassvals)
    print('Reclassifying input point cloud using "{}" with threshold of {}'.format(veg_index, otsu_threshold))
    print('  Original Classes: {}'.format(final_pts))
    # reclassify the original classification values based on the Otsu threshold and reclassification scheme specified
    final_pts[idxvals['vals'][veg_index] > otsu_threshold] = reclasses[1]
    final_pts[idxvals['vals'][veg_index] < otsu_threshold] = reclasses[0]
    print('   UPDATED Classes: {}'.format(final_pts))
    # return the updated classificiation values as an array of numpy integers
    return final_pts.astype(np.int32)


def main():
    # get filenames
    classifyfile = getfile(window_title='Select Point Cloud to Reclassify')
    # classifyfile = 'C:/Users/pwernette/OneDrive - DOI/Machine Learning/test_dat/20180726.laz'

    # get filename
    trained_csv_file = getfile(window_title='Select CSV File Created From vegfilter_training.py')
    # trained_csv_file = 'C:/Users/pwernette/OneDrive - DOI/Machine Learning/test_dat/training_noveg_training_veg.csv'

    # get start time
    startTime = time.time()

    '''
    Read in the complete LAS/LAZ file
    '''
    print('Reading: '+str(classifyfile))
    readtime = time.time()
    f = file.File(str(classifyfile),mode='r')
    print('  Point Cloud to Classify: {}'.format(len(f)))
    print('  Time to read: {}'.format((time.time()-readtime)))

    '''
    Read in training table
    '''
    print('Read in trained info file: {}'.format(trained_csv_file))
    # use index_col=0 since the output file uses the vegetation index as the row name
    veg_otsu_infile = pd.read_csv(trained_csv_file, index_col=0)

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
    rcopy,gcopy,bcopy = deepcopy(f.red),deepcopy(f.green),deepcopy(f.blue)

    '''
    Determine the minimum and maximum values of the r,g,b bands.

    In this case, since we are dealing with point clouds that have 16-bit color depth, the minimum value possible
    is 0 and the maximum value possible is 65,536. However, if the color depth is only 8-bit, then the maximum
    possible color value would be 256.
    '''
    rmin,rmax,gmin,gmax,bmin,bmax = getminmax(rcopy,gcopy,bcopy)

    print('Normalizing R, G, B bands...')
    '''
    Normalize the r,g,b values
    '''
    r = normRGB(normdat(rcopy,rmin,rmax),normdat(rcopy,rmin,rmax),normdat(gcopy,gmin,gmax),normdat(bcopy,bmin,bmax))
    g = normRGB(normdat(gcopy,gmin,gmax),normdat(rcopy,rmin,rmax),normdat(gcopy,gmin,gmax),normdat(bcopy,bmin,bmax))
    b = normRGB(normdat(bcopy,bmin,bmax),normdat(rcopy,rmin,rmax),normdat(gcopy,gmin,gmax),normdat(bcopy,bmin,bmax))
    del(rcopy,gcopy,bcopy)  # clean up the workspace

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
    Apply the threshold from the veg/no-veg binarization to another point cloud using the otsu_appthresh() function.
    '''
    updated_classes = apply_otsu(r,g,b, deepcopy(f.classification), vegetation_index, veg_otsu_infile['threshold'][vegetation_index], reclasses=[2,4])

    '''
    Create copy of entire point cloud, update the classification field, and save the LAS/LAZ file.
    '''
    # create a new file with an updated name ("_reclass_{vegetation_index_name}_veg_noveg.laz" has been appended to the filename)
    f_out = file.File(str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/'), mode='w', header=f.header)
    # copy the points from the origina file to the new output file
    f_out.points = f.points
    # update the classification with the computed/binarized classes from the previous code block
    f_out.classification = updated_classes
    # close the output file
    f_out.close()
    f.close()
    del(r,g,b,updated_classes)

    '''
    Convert the LAS output file to a LAZ compressed file (to save on system storage).
    '''
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
            '''
            Remove the original output LAS file.
            '''
            rmfile = Popen(['rm',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/')])
            rmfile.wait()
            (rmout,rmerr) = rmfile.communicate()
            if rmfile.returncode!=0:
                print(rmerr)
            else:
                print('Successfully deleted old LAS file to save storage space.')
    else:
        print('Successfully converted LAS to LAZ file using 64-bit las2las.')
        '''
        Remove the original output LAS file.
        '''
        rmfile = Popen(['rm',str(ntpath.join(root_dir,(infilename+'_reclass_'+str(vegetation_index)+'_veg_noveg.las'))).replace('\\','/')])
        rmfile.wait()
        (rmout,rmerr) = rmfile.communicate()
        if rmfile.returncode!=0:
            print(rmerr)
        else:
            print('Successfully deleted old LAS file to save storage space.')

    # print complete script runtime
    print('COMPLETE RUNTIME: {}'.format((time.time()-startTime)))

if __name__ == '__main__':
    main()
