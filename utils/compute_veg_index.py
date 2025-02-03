# basic libraries
import os, sys, time, datetime
from datetime import date
import traceback

# import laspy
import laspy
if int(laspy.__version__.split('.')[0]) == 1:
    # in major version laspy 1.x.x files are read using laspy.file.File
    from laspy import file as lf
    
# import libraries for managing and plotting data
import numpy as np
from miscfx import *
from argclass import *


def read_laz(default_args):
    '''
    Read LAS/LAZ point cloud.
    '''
    if default_args.laspy_version == 1:
        try:
            pcloud = lf.File(default_args.file_in, mode='r')
            print('Read {}'.format(default_args.file_in))
        except Exception as e:
            print('ERROR: Unable to read point cloud file(s). See error below for more information.')
            sys.exit(e)
    elif default_args.laspy_version == 2:
        try:
            pcloud = laspy.read(default_args.file_in)
            print('Read {}'.format(default_args.file_in))
        except Exception as e:
            print('ERROR: Unable to read point cloud file(s). See error below for more information.')
            sys.exit(e)
    # las_header = pcloud.header
    return pcloud

def calc_index(las_object, indices=['all'], colordepth=8):
    '''
    Compute specified vegetation indices and/or geometric values.

    Input parameters:
        :param numpy.array lasfileobj: LAS file object
        :param str indices: Vegetation indices to be computed.
            'all' --> all vegetation indices
            'exr' --> extra red index
            'exg' --> extra green index
            'exb' --> extra blue index
            'exgr' --> extra green-red index
            'ngrdi' --> normal red-green difference index
            'mgrvi' --> modified green red vegetation index
            'gli' --> green leaf index
            'rgbvi' --> red green blue vegetation index
            'ikaw' --> Kawashima index
            'gla' --> green leaf algorithm
        :param float geom_radius: Radius used to compute geometric values.

    Returns:
        (1) An n-dimensional array with vegetation index values
            (* denotes optional indices):
            * exr (:py:class:`float`)
            * exg (:py:class:`float`)
            * exb (:py:class:`float`)
            * exgr (:py:class:`float`)
            * ngrdi (:py:class:`float`)
            * mgrvi (:py:class:`float`)
            * gli (:py:class:`float`)
            * rgbvi (:py:class:`float`)
            * ikaw (:py:class:`float`)
            * gla (:py:class:`float`)
        (2) A 1D np.array with vegetation index names

    Notes:
        There is no need to normalize any values before passing a valid
        las or laz file object (from laspy) to this function.
        r, g, and b values are normalized within this updated function (20210801).

    References:
        Excess Red (ExR)
            Meyer, G. E., Hindman, T. W., and Laksmi, K. (1999). Machine vision
            detection parameters for plant species identification. in, eds.
            G. E. Meyer and J. A. DeShazer (Boston, MA), 327–335.
            doi:10.1117/12.336896.
        Excess Green (ExG)
            Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A.
            1995. Color Indices forWeed Identification Under Various Soil,
            Residue, and Lighting Conditions. Trans. ASAE, 38, 259–269.
        Excess Blue (ExB)
            Mao, W.; Wang, Y.;Wang, Y. 2003. Real-time detection of between-row
            weeds using machine vision. In Proceedings of the 2003 ASAE
            Annual Meeting; American Society of Agricultural and Biological
            Engineers, Las Vegas, NV, USA, 27–30 July 2003.
        Excess Green minus R (ExGR)
            Neto, J.C. 2004. A combined statistical-soft computing approach
            for classification and mapping weed species in minimum -tillage
            systems. Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln,
            NE, USA, August 2004.
        Normal Green-Red Difference Index (NGRDI)
            Hunt, E. R., Cavigelli, M., Daughtry, C. S. T., Mcmurtrey, J. E.,
            and Walthall, C. L. (2005). Evaluation of Digital Photography from
            Model Aircraft for Remote Sensing of Crop Biomass and Nitrogen
            Status. Precision Agriculture 6, 359–378.
            doi:10.1007/s11119-005-2324-5.
        Modified Green Red Vegetation Index (MGRVI)
            Tucker, C.J. Red and photographic infrared linear combinations
            for monitoring vegetation. Remote Sens. Environ. 8, 127–150.
        Green Leaf Index (GLI)
            Louhaichi, M.; Borman, M.M.; Johnson, D.E. 2001. Spatially
            located platform and aerial photography for documentation of
            grazing impacts on wheat. Geocarto Int. 16, 65–70.
        Red Green Blue Vegetation Index (RGBVI)
            Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.;
            Broscheit, J.; Gnyp, M.L.; Bareth, G. 2015. Combining UAV-based
            plant height from crop surface models, visible, and near infrared
            vegetation indices for biomass monitoring in barley. Int. J.
            Appl. Earth Obs. Geoinf. 39, 79–87.
        Kawashima Index (IKAW)
            Kawashima, S.; Nakatani, M. 1998. An algorithm for estimating
            chlorophyll content in leaves using a video camera. Ann. Bot.
            81, 49–54.
        Green Leaf Algorithm (GLA)
            Louhaichi, M.; Borman, M.M.; Johnson, D.E. 2001. Spatially
            located platform and aerial photography for documentation of
            grazing impacts on wheat. Geocarto Int. 16, 65–70.
    '''
    # start timer
    start_time = time.time()
    # extract r, g, and b bands from the point cloud
    r,g,b = las_object.red,las_object.green,las_object.blue
    # check for color depth
    if np.amax(r)>256 or np.amax(g)>256 or np.amax(b)>256:
        colordepth = 16
    # normalize R, G, and B bands
    # NOTE: includes conversion of r, g, b to np.float32 data type
    r,g,b = normBands(r,g,b, depth=colordepth)
    # use n-dimensional numpy array as a data container for
    #  1) output values
    #  2) output attribute names
    # pdindex = np.empty(shape=(0,(len(r[0]))), dtype=np.float32)
    # pdindexnames = np.empty(shape=(0,0))

    # vegidx = {'red': r,
    #           'green': g,
    #           'blue': b,
    #           'X': las_object.X,
    #           'Y': las_object.Y,
    #           'Z': las_object.Z}

    # pdindexnames = np.append(pdindexnames, 'r')
    # pdindexnames = np.append(pdindexnames, 'g')
    # pdindexnames = np.append(pdindexnames, 'b')

    # # retain xyz coordinates
    # pdindex = np.append(pdindex, [las_object.X], axis=0)
    # pdindexnames = np.append(pdindexnames, 'x')
    # pdindex = np.append(pdindex, [las_object.Y], axis=0)
    # pdindexnames = np.append(pdindexnames, 'y')
    # pdindex = np.append(pdindex, [las_object.Z], axis=0)
    # pdindexnames = np.append(pdindexnames, 'Z')
    '''
    Compute vegetation indices based on user specifications

    for each index, the following steps are taken:
        1) compute the index from R, G, and/or B normalized values
        2) append the vegetation index values to the output
            multidimensional array
        3) append the vegetation index name to the output array
    '''
    if ('all' in indices) or ('exr' in indices):
        # exr = 1.4*b-g
        
        # vegidx['exr'] = 1.4*b-g
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="exr",
            type=np.float32,
            description="exr"
            ))
        las_object.exr = 1.4*b-g

        # pdindex = np.append(pdindex, exr, axis=0)
        # pdindexnames = np.append(pdindexnames, 'exr')
    if ('all' in indices) or ('exg' in indices):
        # exg = 2*g-r-b

        # vegidx['exg'] = 2*g-r-b
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="exg",
            type=np.float32,
            description="exg"
            ))
        las_object.exg = 2*g-r-b

        # pdindex = np.append(pdindex, exg, axis=0)
        # pdindexnames = np.append(pdindexnames, 'exg')
    if ('all' in indices) or ('exb' in indices):
        # exb = 1.4*r-g

        # vegidx['exb'] = 1.4*r-g
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="exb",
            type=np.float32,
            description="exb"
            ))
        las_object.exb = 1.4*r-g

        # pdindex = np.append(pdindex, exb, axis=0)
        # del(exb)
        # pdindexnames = np.append(pdindexnames, 'exb')
    if ('all' in indices) or ('exgr' in indices):
        # exgr = exg-exr

        # vegidx['exgr'] = vegidx['exg']-vegidx['exr']
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="exgr",
            type=np.float32,
            description="exgr"
            ))
        las_object.exgr = (2*g-r-b)-(1.4*b-g)

        # pdindex = np.append(pdindex, exgr, axis=0)
        # del(exg,exr,exgr)
        # pdindexnames = np.append(pdindexnames, 'exgr')
    if ('all' in indices) or ('ngrdi' in indices):
        # ngrdi = np.divide((g-r), (r+g),
        #                   out=np.zeros_like((g-r)),
        #                   where=(g+r)!=np.zeros_like(g))

        # vegidx['ngrdi'] = np.divide((g-r), (r+g),
        #                   out=np.zeros_like((g-r)),
        #                   where=(g+r)!=np.zeros_like(g))
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="ngrdi",
            type=np.float32,
            description="ngrdi"
            ))
        las_object.ngrdi = np.divide((g-r), (r+g),
                          out=np.zeros_like((g-r)),
                          where=(g+r)!=np.zeros_like(g))
        
        # pdindex = np.append(pdindex, ngrdi, axis=0)
        # del(ngrdi)
        # pdindexnames = np.append(pdindexnames, 'ngrdi')
    if ('all' in indices) or ('mgrvi' in indices):
        # mgrvi = np.divide((np.power(g,2)-np.power(r,2)),
        #                   (np.power(r,2)+np.power(g,2)),
        #                   out=np.zeros_like((g-r)),
        #                   where=(np.power(g,2)+np.power(r,2))!=0)

        # vegidx['mgrvi'] = np.divide((np.power(g,2)-np.power(r,2)),
        #                   (np.power(r,2)+np.power(g,2)),
        #                   out=np.zeros_like((g-r)),
        #                   where=(np.power(g,2)+np.power(r,2))!=0)
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="mgrvi",
            type=np.float32,
            description="mgrvi"
            ))
        las_object.mgrvi = np.divide((np.power(g,2)-np.power(r,2)),
                          (np.power(r,2)+np.power(g,2)),
                          out=np.zeros_like((g-r)),
                          where=(np.power(g,2)+np.power(r,2))!=0)
        
        # pdindex = np.append(pdindex, mgrvi, axis=0)
        # del(mgrvi)
        # pdindexnames = np.append(pdindexnames, 'mgrvi')
    if ('all' in indices) or ('gli' in indices):
        # gli = np.divide(2*g-r-b, 2*g+r+b,
        #                 out=np.zeros_like((2*g-r-b)),
        #                 where=(2*g+r+b)!=0)

        # vegidx['gli'] = np.divide(2*g-r-b, 2*g+r+b,
        #                 out=np.zeros_like((2*g-r-b)),
        #                 where=(2*g+r+b)!=0)
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="gli",
            type=np.float32,
            description="gli"
            ))
        las_object.gli = np.divide(2*g-r-b, 2*g+r+b,
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        
        # pdindex = np.append(pdindex, gli, axis=0)
        # del(gli)
        # pdindexnames = np.append(pdindexnames, 'gli')
    if ('all' in indices) or ('rgbvi' in indices):
        # rgbvi = np.divide((np.power(g,2)-b*r),
        #                   (np.power(g,2)+b*r),
        #                   out=np.zeros_like((np.power(g,2)-b*r)),
        #                   where=(np.power(g,2)+b*r)!=0)

        # vegidx['rgbvi'] = np.divide((np.power(g,2)-b*r),
        #                   (np.power(g,2)+b*r),
        #                   out=np.zeros_like((np.power(g,2)-b*r)),
        #                   where=(np.power(g,2)+b*r)!=0)
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="rgbvi",
            type=np.float32,
            description="rgbvi"
            ))
        las_object.rgbvi = np.divide((np.power(g,2)-b*r),
                          (np.power(g,2)+b*r),
                          out=np.zeros_like((np.power(g,2)-b*r)),
                          where=(np.power(g,2)+b*r)!=0)
        
        # pdindex = np.append(pdindex, rgbvi, axis=0)
        # del(rgbvi)
        # pdindexnames = np.append(pdindexnames, 'rgbvi')
    if ('all' in indices) or ('ikaw' in indices):
        # ikaw = np.divide((r-b), (r+b),
        #                  out=np.zeros_like((r-b)),
        #                  where=(r+b)!=0)

        # vegidx['ikaw'] = np.divide((r-b), (r+b),
        #                  out=np.zeros_like((r-b)),
        #                  where=(r+b)!=0)
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="ikaw",
            type=np.float32,
            description="ikaw"
            ))
        las_object.ikaw = np.divide((r-b), (r+b),
                         out=np.zeros_like((r-b)),
                         where=(r+b)!=0)
        
        # pdindex = np.append(pdindex, ikaw, axis=0)
        # del(ikaw)
        # pdindexnames = np.append(pdindexnames, 'ikaw')
    if ('all' in indices) or ('gla' in indices):
        # gla = np.divide((2*g-r-b), (2*g+r+b),
        #                 out=np.zeros_like((2*g-r-b)),
        #                 where=(2*g+r+b)!=0)

        # vegidx['gla'] = np.divide((2*g-r-b), (2*g+r+b),
        #                 out=np.zeros_like((2*g-r-b)),
        #                 where=(2*g+r+b)!=0)
        las_object.add_extra_dim(laspy.ExtraBytesParams(
            name="gla",
            type=np.float32,
            description="gla"
            ))
        las_object.gla = np.divide((2*g-r-b), (2*g+r+b),
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        
        # pdindex = np.append(pdindex, gla, axis=0)
        # del(gla)
        # pdindexnames = np.append(pdindexnames, 'gla')
    '''
    Use vegetation indices names index to generate a dictionary
    (indexnamesdict) that can be used to reference the appropriate
    index of the data array.

    Example:
        Instead of using pdindex[0] and having to remember that the
        0 index represents the X coordinates (scaled), use the
        following to access the X coordinates:
            pdindex[indexnames['x']]
    '''
    # print('Values and indices: {}'.format(pdindexnames))
    # print('    Computation time: {}s'.format((time.time()-start_time)))
    # return pdindexnames,pdindex
    
    return las_object

def segment_point_cloud(input_point_cloud, vegetation_index, vegetation_threshold):
    print('{} points exceed threshold value ({}) for index ({})'.format(len(input_point_cloud.classification[input_point_cloud[vegetation_index] >= vegetation_threshold]), vegetation_threshold, vegetation_index))
    input_point_cloud.classification[input_point_cloud[vegetation_index] >= vegetation_threshold] = 4
    # outdat_pred_reclass[(outdat_pred_reclass >= threshold_val)] = 4  # reclass veg. points
    # outdat_pred_reclass[(outdat_pred_reclass < threshold_val)] = 2   # reclass no veg. points
    # outdat_pred_reclass = outdat_pred_reclass.flatten().astype(np.int32)  # convert float to int

def main(default_values):
    # get and update laspy version
    print("laspy Information:")
    # print laspy version installed and configured
    print("   laspy Version: {}\n".format(laspy.__version__))

    default_values.laspy_version = int(laspy.__version__.split('.')[0])
    if default_values.laspy_version == 1:
        # in major version laspy 1.x.x files are read using laspy.file.File
        from laspy import file as lf
    elif default_values.laspy_version == 2:
        import lazrs
    else:
        print('\nERROR: laspy has unsupported major version = {}'.format(default_values.laspy_version))
        sys.exit()

    # read point cloud
    dat = read_laz(default_values)

    las_obj = calc_index(las_object=dat, 
                         indices=default_values.veg_index)
    
    # get root directory
    default_values.rootdir = os.path.split(default_values.file_in)[0]
    
    if not 'all' in default_values.veg_index:
        print('{} points exceed threshold value ({}) for index ({})'.format(len(las_obj['classification'][las_obj[default_values.veg_index] >= float(default_values.veg_threshold)]), default_values.veg_threshold, default_values.veg_index))
        
        if 'exb' in default_values.veg_index:
            las_obj['classification'][las_obj[default_values.veg_index] <= float(default_values.veg_threshold)] = 4
        else:
            las_obj['classification'][las_obj[default_values.veg_index] >= float(default_values.veg_threshold)] = 4
        
        default_values.file_out = os.path.join(default_values.rootdir,os.path.basename(default_values.file_in).replace('.laz', '_'+str(default_values.veg_index)+'_'+str(default_values.veg_threshold)+'.laz'))
        print('\nUpdated output filename to: {}'.format(default_values.file_out))
    else:
        default_values.file_out = os.path.join(default_values.rootdir,os.path.basename(default_values.file_in).replace('.laz', '_all.laz'))
        print('\nUpdated output filename to: {}'.format(default_values.file_out))
    
    # write out LAZ file
    print('\nWriting new LAZ file to:\n  {}'.format(default_values.file_out))
    las_obj.write(default_values.file_out)

if __name__ == '__main__':
    defs = Args('defs')
    defs.parse_cmd_arguments()

    if defs.gui:
        foo = App()
        foo.create_widgets(defs)
        foo.mainloop()
    
    defs.laspy_version = 0

    # try:
    main(default_values=defs)
    # except:
    #     traceback.print_exc()