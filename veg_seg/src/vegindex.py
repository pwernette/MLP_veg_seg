import sys
import time
import numpy as np
from sklearn import preprocessing as skpre
from .miscfx import *

def vegidx(lasfileobj, indices=['rgb'], geom_metrics=[], colordepth=8, geom_radius=1.00, xyz_mins=[0,0,0], xyz_maxs=[0,0,0]):
    '''
    Compute specified vegetation indices and/or geometric values.

    Input parameters:
        :param numpy.array lasfileobj: LAS file object
        :param str indices: Vegetation indices to be computed.
            'rgb' --> only RGB values (no veg indices or geom values)
            'coords' --> include XYZ coordinates in output
            'simple' --> RGB values and ExR, ExG, ExB, ExGR veg indices
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
        (1) Original point cloud with vegetation index values added as extra bytes
            (* denotes optional indices):
            * x (:py:class:`float`)
            * y (:py:class:`float`)
            * z (:py:class:`float`)
              r (:py:class:`float`)
              g (:py:class:`float`)
              b (:py:class:`float`)
            * sd (:py:class:`float`)
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
    dim_names = list(lasfileobj.point_format.dimension_names)

    # start timer
    start_time = time.time()

    # check that red, green, and blue data is present
    assert 'red' in dim_names
    assert 'green' in dim_names
    assert 'blue' in dim_names

    # extract r, g, and b bands from the point cloud
    r,g,b = lasfileobj.red,lasfileobj.green,lasfileobj.blue

    # check for color depth
    if np.amax(r)>256 or np.amax(g)>256 or np.amax(b)>256:
        colordepth = 16
    ''' normalize R, G, and B bands '''
    # add extra bytes fields for normalized RGB
    if not 'rnorm' in dim_names:
        lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="rnorm",
                type=np.float32,
                description="red_normalized"
                ))
    if not 'gnorm' in dim_names:
        lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="gnorm",
                type=np.float32,
                description="green_normalized"
                ))
    if not 'bnorm' in dim_names:
        lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="bnorm",
                type=np.float32,
                description="blue_normalized"
                ))
    # NOTE: includes conversion of r, g, b to np.float32 data type
    lasfileobj.rnorm,lasfileobj.gnorm,lasfileobj.bnorm = normBands(r,g,b, depth=colordepth)
    
    if any('xyz' in i for i in indices):
        print('Adding X,Y,Z Normalized Values')
        lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="x_norm",
                type=np.float32,
                description="x_norm"
                ))
        min_max_scaler = skpre.MinMaxScaler()
        if xyz_mins == [0,0,0]:
            lasfileobj.x_norm = min_max_scaler.fit_transform(np.array(lasfileobj.x).reshape(-1,1)).flatten()
        else:
            min_max_scaler.fit(np.array([xyz_mins[0], xyz_maxs[0]]).reshape(-1,1))
            lasfileobj.x_norm = min_max_scaler.transform(np.array(lasfileobj.x).reshape(-1,1)).flatten()

        lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="y_norm",
                type=np.float32,
                description="y_norm"
                ))
        min_max_scaler = skpre.MinMaxScaler()
        if xyz_mins == [0,0,0]:
            lasfileobj.y_norm = min_max_scaler.fit_transform(np.array(lasfileobj.y).reshape(-1,1)).flatten()
        else:
            min_max_scaler.fit(np.array([xyz_mins[1], xyz_maxs[1]]).reshape(-1,1))
            lasfileobj.y_norm = min_max_scaler.transform(np.array(lasfileobj.y).reshape(-1,1)).flatten()

        lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="z_norm",
                type=np.float32,
                description="z_norm"
                ))
        min_max_scaler = skpre.MinMaxScaler()
        if xyz_mins == [0,0,0]:
            lasfileobj.z_norm = min_max_scaler.fit_transform(np.array(lasfileobj.z).reshape(-1,1)).flatten()
        else:
            min_max_scaler.fit(np.array([xyz_mins[2], xyz_maxs[2]]).reshape(-1,1))
            lasfileobj.z_norm = min_max_scaler.transform(np.array(lasfileobj.z).reshape(-1,1)).flatten()

    '''
    Compute vegetation indices based on user specifications

    for each index, the following steps are taken:
        1) compute the index from R, G, and/or B normalized values
        2) append the vegetation index values to the output
            multidimensional array
        3) append the vegetation index name to the output array
    '''
    if ('all' in indices) or ('exr' in indices) or ('simple' in indices):
        if not 'exr' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="exr",
                type=np.float32,
                description="exr"
                ))
        lasfileobj.exr = 1.4*lasfileobj.bnorm-lasfileobj.gnorm
    if ('all' in indices) or ('exg' in indices) or ('simple' in indices):
        if not 'exg' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="exg",
                type=np.float32,
                description="exg"
                ))
        lasfileobj.exg = 2*lasfileobj.gnorm-lasfileobj.rnorm-lasfileobj.bnorm
    if ('all' in indices) or ('exb' in indices) or ('simple' in indices):
        if not 'exb' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="exb",
                type=np.float32,
                description="exb"
                ))
        lasfileobj.exb = 1.4*r-g
    if ('all' in indices) or ('exgr' in indices) or ('simple' in indices):
        if not 'exgr' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="exgr",
                type=np.float32,
                description="exgr"
                ))
        lasfileobj.exgr = (2*lasfileobj.gnorm-lasfileobj.rnorm-lasfileobj.bnorm)-(1.4*lasfileobj.bnorm-lasfileobj.gnorm)
    if ('all' in indices) or ('ngrdi' in indices):
        if not 'ngrdi' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="ngrdi",
                type=np.float32,
                description="ngrdi"
                ))
        lasfileobj.ngrdi = np.divide((lasfileobj.gnorm-lasfileobj.rnorm), (lasfileobj.rnorm+lasfileobj.gnorm),
                          out=np.zeros_like((lasfileobj.gnorm-lasfileobj.rnorm)),
                          where=(lasfileobj.gnorm+lasfileobj.rnorm)!=np.zeros_like(lasfileobj.gnorm))
    if ('all' in indices) or ('mgrvi' in indices):
        if not 'mgrvi' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="mgrvi",
                type=np.float32,
                description="mgrvi"
                ))
        lasfileobj.mgrvi = np.divide((np.power(lasfileobj.gnorm,2)-np.power(lasfileobj.rnorm,2)),
                          (np.power(lasfileobj.rnorm,2)+np.power(lasfileobj.gnorm,2)),
                          out=np.zeros_like((lasfileobj.gnorm-lasfileobj.rnorm)),
                          where=(np.power(lasfileobj.gnorm,2)+np.power(lasfileobj.rnorm,2))!=0)
    if ('all' in indices) or ('gli' in indices):
        if not 'gli' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="gli",
                type=np.float32,
                description="gli"
                ))
        lasfileobj.gli = np.divide(2*lasfileobj.gnorm-lasfileobj.rnorm-lasfileobj.bnorm, 2*lasfileobj.gnorm+lasfileobj.rnorm+lasfileobj.bnorm,
                        out=np.zeros_like((2*lasfileobj.gnorm-lasfileobj.rnorm-lasfileobj.bnorm)),
                        where=(2*lasfileobj.gnorm+lasfileobj.rnorm+lasfileobj.bnorm)!=0)
    if ('all' in indices) or ('rgbvi' in indices):
        if not 'rgbvi' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="rgbvi",
                type=np.float32,
                description="rgbvi"
                ))
        lasfileobj.rgbvi = np.divide((np.power(lasfileobj.gnorm,2)-lasfileobj.bnorm*lasfileobj.rnorm),
                          (np.power(lasfileobj.gnorm,2)+lasfileobj.bnorm*lasfileobj.rnorm),
                          out=np.zeros_like((np.power(lasfileobj.gnorm,2)-lasfileobj.bnorm*lasfileobj.rnorm)),
                          where=(np.power(lasfileobj.gnorm,2)+lasfileobj.bnorm*lasfileobj.rnorm)!=0)
    if ('all' in indices) or ('ikaw' in indices):
        if not 'ikaw' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="ikaw",
                type=np.float32,
                description="ikaw"
                ))
        lasfileobj.ikaw = np.divide((lasfileobj.rnorm-lasfileobj.bnorm), (lasfileobj.rnorm+lasfileobj.bnorm),
                         out=np.zeros_like((lasfileobj.rnorm-lasfileobj.bnorm)),
                         where=(lasfileobj.rnorm+lasfileobj.bnorm)!=0)
    if ('all' in indices) or ('gla' in indices):
        if not 'gla' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="gla",
                type=np.float32,
                description="gla"
                ))
        lasfileobj.gla = np.divide((2*lasfileobj.gnorm-lasfileobj.rnorm-lasfileobj.bnorm), (2*lasfileobj.gnorm+lasfileobj.rnorm+lasfileobj.bnorm),
                        out=np.zeros_like((2*lasfileobj.gnorm-lasfileobj.rnorm-lasfileobj.bnorm)),
                        where=(2*lasfileobj.gnorm+lasfileobj.rnorm+lasfileobj.bnorm)!=0)
        
    # if standard deviation is specified as a geometric metric, then
    # compute the sd using a pointwise approach
    # NOTE: This computation is very time and resource expensive.
    if any('sd' in m for m in geom_metrics) or any('sd' in m for m in indices):
        print('    Calculating standard deviation in X, Y, and Z')
        # compute standard deviations
        starttime = time.time()
        if not 'sd_x' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="sd_x",
                type=np.float32,
                description="standard_deviation_x_direction"
                ))
        if not 'sd_y' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="sd_y",
                type=np.float32,
                description="standard_deviation_y_direction"
                ))
        if not 'sd_z' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="sd_z",
                type=np.float32,
                description="standard_deviation_z_direction"
                ))
        lasfileobj.sd_x,lasfileobj.sd_y,lasfileobj.sd_z = calc_sd(np.array([lasfileobj.x, lasfileobj.y, lasfileobj.z]).transpose(), rad=geom_radius)
        # # normalize the values
        # mmscaler = skpre.MinMaxScaler()
        # lasfileobj.sd_x = mmscaler.fit_transform(lasfileobj.sd_x)
        # lasfileobj.sd_y = mmscaler.fit_transform(lasfileobj.sd_y)
        # lasfileobj.sd_z = mmscaler.fit_transform(lasfileobj.sd_z)
        print("Time to compute X,Y,Z Standard Deviations = {}".format(time.time()-starttime))
    if any('3d' in m for m in geom_metrics) or any('3d' in m for m in indices):
        print('    Calculating 3D standard deviation')
        # compute 3D standard deviation
        starttime = time.time()
        if not 'sd3d' in dim_names:
            lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
                name="sd3d",
                type=np.float32,
                description="standard_deviation"
                ))
        lasfileobj.sd3d = calc_3d_sd(np.array([lasfileobj.x, lasfileobj.y, lasfileobj.z]).transpose(), rad=geom_radius)
        print("Time to compute 3D Standard Deviation = {}".format(time.time()-starttime))
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
    print('Computed indices: {}'.format(indices))
    print('  Computation time: {}s'.format((time.time()-start_time)))
    
    return lasfileobj


def veg_rgb(lasfileobj, colordepth=8):
    '''
    Compute specified vegetation indices and/or geometric values.

    Input parameters:
        :param numpy.array lasfileobj: LAS file object

    Returns:
        (1) An n-dimensional array with RGB values
              r (:py:class:`float`)
              g (:py:class:`float`)
              b (:py:class:`float`)
        (2) A 1D np.array with RGB names

    Notes:
        There is no need to normalize any values before passing a valid
        las or laz file object (from laspy) to this function.
        r, g, and b values are normalized within this updated function (20210801).
    '''
    dim_names = list(lasfileobj.point_format.dimension_names)

    # start timer
    start_time = time.time()

    # check that red, green, and blue data is present
    assert 'red' in dim_names
    assert 'green' in dim_names
    assert 'blue' in dim_names

    # extract r, g, and b bands from the point cloud
    r,g,b = lasfileobj.red,lasfileobj.green,lasfileobj.blue
    # check for color depth
    if np.amax(r)>256 or np.amax(g)>256 or np.amax(b)>256:
        colordepth = 16
    ''' normalize R, G, and B bands '''
    # add extra bytes fields for normalized RGB
    lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
            name="rnorm",
            type=np.float32,
            description="red_normalized"
            ))
    lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
            name="gnorm",
            type=np.float32,
            description="green_normalized"
            ))
    lasfileobj.add_extra_dim(laspy.ExtraBytesParams(
            name="bnorm",
            type=np.float32,
            description="blue_normalized"
            ))
    # NOTE: includes conversion of r, g, b to np.float32 data type
    lasfileobj.rnorm,lasfileobj.gnorm,lasfileobj.bnorm = normBands(lasfileobj.red,lasfileobj.green,lasfileobj.blue, depth=colordepth)

    return lasfileobj