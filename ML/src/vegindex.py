import sys
import time
import numpy as np
from .miscfx import *

def vegidx(lasfileobj, geom_metrics=[], indices=['rgb'], colordepth=8, geom_radius=0.10):
    '''
    Compute specified vegetation indices and/or geometric values.

    Input parameters:
        :param numpy.array lasfileobj: LAS file object
        :param numpy.array r: Array of normalized red values
        :param numpy.array g: Array of normalized green values
        :param numpy.array b: Array of normalized blue values
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
              x (:py:class:`float`)
              y (:py:class:`float`)
              z (:py:class:`float`)
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
        (2) A 1D np.array with vegetation index names

    Notes:
        There is no need to normalize any values before passing a valid
        las or laz file object (from laspy) to this function.
        r, g, and b values are normalized within this updated function (20210801).

    References:
        Excess Red (ExR)
            Meyer, G.E.; Neto, J.C. 2008. Verification of color vegetation
            indices for automated crop imaging applications. Comput.
            Electron. Agric. 63, 282–293.
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
            Tucker, C.J. Red and photographic infrared linear combinations
            for monitoring vegetation. Remote Sens. Environ. 8, 127–150.
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
    r,g,b = lasfileobj.red,lasfileobj.green,lasfileobj.blue
    # check for color depth
    if np.amax(r)>256 or np.amax(g)>256 or np.amax(b)>256:
        colordepth = 16
    # normalize R, G, and B bands
    # NOTE: includes conversion of r, g, b to np.float32 data type
    r,g,b = normBands(r,g,b, depth=colordepth)
    # use n-dimensional numpy array as a data container for
    #  1) output values
    #  2) output attribute names
    pdindex = np.empty(shape=(0,(len(r[0]))), dtype=np.float32)
    pdindexnames = np.empty(shape=(0,0))
    # scale x, y, and z coordinates
    if ('x' in indices) or ('sd' in indices) or ('coords' in indices) or ('all' in indices):
        xs = scale_dims(lasfileobj)[0]
        pdindex = np.append(pdindex, [xs], axis=0)
        pdindexnames = np.append(pdindexnames, 'x')
    if ('y' in indices) or ('sd' in indices) or ('coords' in indices) or ('all' in indices):
        ys = scale_dims(lasfileobj)[1]
        pdindex = np.append(pdindex, [ys], axis=0)
        pdindexnames = np.append(pdindexnames, 'y')
    if ('z' in indices) or ('sd' in indices) or ('coords' in indices) or ('all' in indices):
        zs = scale_dims(lasfileobj)[2]
        pdindex = np.append(pdindex, [zs], axis=0)
        pdindexnames = np.append(pdindexnames, 'z')
    # if standard deviation is specified as a geometric metric, then
    # compute the sd using a pointwise approach
    # NOTE: This computation is very time and resource expensive.
    if 'sd' in geom_metrics:
        # compute 3D standard deviation
        starttime = time.time()
        sd3d = calc_3d_sd(np.array([xs, ys, zs]).transpose(), rad=0.10)
        print("Time to compute SD = {}".format(time.time()-starttime))
        # append 3D sd to output array
        pdindex = np.append(pdindex, [sd3d], axis=0)
        # clean up workspace memory
        del(xs,ys,zs,sd3d)
        pdindexnames = np.append(pdindexnames, 'sd')
    # since not all users may require R, G, B values, these are optional but
    # default variables
    if ('r' in indices) or ('rgb' in indices) or ('all' in indices):
        pdindex = np.append(pdindex, r, axis=0)
        pdindexnames = np.append(pdindexnames, 'r')
    if ('g' in indices) or ('rgb' in indices) or ('all' in indices):
        pdindex = np.append(pdindex, g, axis=0)
        pdindexnames = np.append(pdindexnames, 'g')
    if ('b' in indices) or ('rgb' in indices) or ('all' in indices):
        pdindex = np.append(pdindex, b, axis=0)
        pdindexnames = np.append(pdindexnames, 'b')
    # option to include intensty as a variable
    if ('intensity' in indices) or ('all' in indices):
        pdindex = np.append(pdindex, lasfileobj.intensity, axis=0)
        pdindexnames = np.append(pdindexnames, 'intensity')
    '''
    Compute vegetation indices based on user specifications

    for each index, the following steps are taken:
        1) compute the index from R, G, and/or B normalized values
        2) append the vegetation index values to the output
            multidimensional array
        3) append the vegetation index name to the output array
    '''
    if ('all' in indices) or ('exr' in indices) or ('exgr' in indices) or ('simple' in indices):
        exr = 1.4*b-g
        pdindex = np.append(pdindex, exr, axis=0)
        pdindexnames = np.append(pdindexnames, 'exr')
    if ('all' in indices) or ('exg' in indices) or ('exgr' in indices) or ('simple' in indices):
        exg = 2*g-r-b
        pdindex = np.append(pdindex, exg, axis=0)
        pdindexnames = np.append(pdindexnames, 'exg')
    if ('all' in indices) or ('exb' in indices) or ('simple' in indices):
        exb = 1.4*r-g
        pdindex = np.append(pdindex, exb, axis=0)
        del(exb)
        pdindexnames = np.append(pdindexnames, 'exb')
    if ('all' in indices) or ('exgr' in indices) or ('simple' in indices):
        exgr = exg-exr
        pdindex = np.append(pdindex, exgr, axis=0)
        del(exg,exr,exgr)
        pdindexnames = np.append(pdindexnames, 'exgr')
    if ('all' in indices) or ('ngrdi' in indices):
        ngrdi = np.divide((g-r), (r+g),
                          out=np.zeros_like((g-r)),
                          where=(g+r)!=np.zeros_like(g))
        pdindex = np.append(pdindex, ngrdi, axis=0)
        del(ngrdi)
        pdindexnames = np.append(pdindexnames, 'ngrdi')
    if ('all' in indices) or ('mgrvi' in indices):
        mgrvi = np.divide((np.power(g,2)-np.power(r,2)),
                          (np.power(r,2)+np.power(g,2)),
                          out=np.zeros_like((g-r)),
                          where=(np.power(g,2)+np.power(r,2))!=0)
        pdindex = np.append(pdindex, mgrvi, axis=0)
        del(mgrvi)
        pdindexnames = np.append(pdindexnames, 'mgrvi')
    if ('all' in indices) or ('gli' in indices):
        gli = np.divide(2*g-r-b, 2*g+r+b,
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        pdindex = np.append(pdindex, gli, axis=0)
        del(gli)
        pdindexnames = np.append(pdindexnames, 'gli')
    if ('all' in indices) or ('rgbvi' in indices):
        rgbvi = np.divide((np.power(g,2)-b*r),
                          (np.power(g,2)+b*r),
                          out=np.zeros_like((np.power(g,2)-b*r)),
                          where=(np.power(g,2)+b*r)!=0)
        pdindex = np.append(pdindex, rgbvi, axis=0)
        del(rgbvi)
        pdindexnames = np.append(pdindexnames, 'rgbvi')
    if ('all' in indices) or ('ikaw' in indices):
        ikaw = np.divide((r-b), (r+b),
                         out=np.zeros_like((r-b)),
                         where=(r+b)!=0)
        pdindex = np.append(pdindex, ikaw, axis=0)
        del(ikaw)
        pdindexnames = np.append(pdindexnames, 'ikaw')
    if ('all' in indices) or ('gla' in indices):
        gla = np.divide((2*g-r-b), (2*g+r+b),
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        pdindex = np.append(pdindex, gla, axis=0)
        del(gla)
        pdindexnames = np.append(pdindexnames, 'gla')
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
    print('Computed indices: {}'.format(pdindexnames))
    print('  Computation time: {}s'.format((time.time()-start_time)))
    return pdindexnames,pdindex
#     indexnamesdict = dict((j,i) for i,j in enumerate(pdindexnames))
#     return indexnamesdict,pdindex
