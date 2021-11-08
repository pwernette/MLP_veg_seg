import sys, time
import numpy as np
from .functions import *

def vegidx(lasfileobj, geom_metrics=[], indices=[], colordepth=8, geom_radius=0.10):
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
        An n-dimensional array (* denotes optional indices):
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

    Notes:
        There is no need to normalize any values before passing a valid
        las or laz file object (from laspy) to this function.
        r, g, and b values are normalized within this updated function (20210801).

    References:
        Excess Red (ExR)
               Meyer, G.E.; Neto, J.C. Verification of color vegetation indices for automated crop imaging applications.
               Comput. Electron. Agric. 2008, 63, 282–293.
        Excess Green (ExG)
               Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color Indices forWeed Identification Under
               Various Soil, Residue, and Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
        Excess Blue (ExB)
               Mao,W.;Wang, Y.;Wang, Y. Real-time detection of between-row weeds using machine vision. In Proceedings
               of the 2003 ASAE Annual Meeting; American Society of Agricultural and Biological Engineers, Las Vegas,
               NV, USA, 27–30 July 2003.
        Excess Green minus R (ExGR)
               Neto, J.C. A combined statistical-soft computing approach for classification and mapping weed species in
               minimum -tillage systems. Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln, NE, USA, August 2004.
        Normal Green-Red Difference Index (NGRDI)
               Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
               Environ. 1979, 8, 127–150.
        Modified Green Red Vegetation Index (MGRVI)
               Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens.
               Environ. 1979, 8, 127–150.
        Green Leaf Index (GLI)
               Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
               documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
        Red Green Blue Vegetation Index (RGBVI)
               Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.; Broscheit, J.; Gnyp, M.L.; Bareth, G. Combining
               UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass
               monitoring in barley. Int. J. Appl. Earth Obs. Geoinf. 2015, 39, 79–87.
        Kawashima Index (IKAW)
               Kawashima, S.; Nakatani, M. An algorithm for estimating chlorophyll content in leaves using a video camera.
               Ann. Bot. 1998, 81, 49–54.
        Green Leaf Algorithm (GLA)
               Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform and aerial photography for
               documentation of grazing impacts on wheat. Geocarto Int. 2001, 16, 65–70.
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
    pdindex = np.empty(shape=(r.shape[1]), dtype=np.float32)
    pdindexnames = np.empty(shape=(0,0))
    minarr = np.empty(shape=(0,0))
    maxarr = np.empty(shape=(0,0))
    # scale x, y, and z coordinates
    xs,ys,zs = scale_dims(lasfileobj)
    # ys = scale_dims(lasfileobj)[1]
    # zs = scale_dims(lasfileobj)[2]
    # append x, y, and z(raw values) to the output n-dim numpy array
    pdindex = np.vstack((pdindex, xs))
    pdindex = np.vstack((pdindex, ys))
    pdindex = np.vstack((pdindex, zs))
    # clean up the workspace (save memory)
    pdindexnames = np.append(pdindexnames, ['x','y','z'])
    pdindex = np.vstack((pdindex, r))
    pdindex = np.vstack((pdindex, g))
    pdindex = np.vstack((pdindex, b))
    # append index names to names array
    pdindexnames = np.append(pdindexnames, ['r','g','b'])
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
        pdindex = np.vstack((pdindex, exr))
        pdindexnames = np.append(pdindexnames, 'exr')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.4)
    if ('all' in indices) or ('exg' in indices) or ('exgr' in indices) or ('simple' in indices):
        exg = 2*g-r-b
        pdindex = np.vstack((pdindex, exg))
        pdindexnames = np.append(pdindexnames, 'exg')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 2.0)
    if ('all' in indices) or ('exb' in indices) or ('simple' in indices):
        exb = 1.4*r-g
        pdindex = np.vstack((pdindex, exb))
        del(exb)
        pdindexnames = np.append(pdindexnames, 'exb')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.4)
    if ('all' in indices) or ('exgr' in indices) or ('simple' in indices):
        exgr = exg-exr
        pdindex = np.vstack((pdindex, exgr))
        del(exg,exr,exgr)
        pdindexnames = np.append(pdindexnames, 'exgr')
        maxarr = np.append(maxarr, -2.4)
        maxarr = np.append(maxarr, 3.0)
    if ('all' in indices) or ('ngrdi' in indices):
        ngrdi = np.divide((g-r), (r+g),
                          out=np.zeros_like((g-r)),
                          where=(g+r)!=np.zeros_like(g))
        pdindex = np.vstack((pdindex, ngrdi))
        del(ngrdi)
        pdindexnames = np.append(pdindexnames, 'ngrdi')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.0)
    if ('all' in indices) or ('mgrvi' in indices):
        mgrvi = np.divide((np.power(g,2)-np.power(r,2)),
                          (np.power(r,2)+np.power(g,2)),
                          out=np.zeros_like((g-r)),
                          where=(np.power(g,2)+np.power(r,2))!=0)
        pdindex = np.vstack((pdindex, mgrvi))
        del(mgrvi)
        pdindexnames = np.append(pdindexnames, 'mgrvi')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.0)
    if ('all' in indices) or ('gli' in indices):
        gli = np.divide(2*g-r-b, 2*g+r+b,
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        pdindex = np.vstack((pdindex, gli))
        del(gli)
        pdindexnames = np.append(pdindexnames, 'gli')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.0)
    if ('all' in indices) or ('rgbvi' in indices):
        rgbvi = np.divide((np.power(g,2)-b*r),
                          (np.power(g,2)+b*r),
                          out=np.zeros_like((np.power(g,2)-b*r)),
                          where=(np.power(g,2)+b*r)!=0)
        pdindex = np.vstack((pdindex, rgbvi))
        del(rgbvi)
        pdindexnames = np.append(pdindexnames, 'rgbvi')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.0)
    if ('all' in indices) or ('ikaw' in indices):
        ikaw = np.divide((r-b), (r+b),
                         out=np.zeros_like((r-b)),
                         where=(r+b)!=0)
        pdindex = np.vstack((pdindex, ikaw))
        del(ikaw)
        pdindexnames = np.append(pdindexnames, 'ikaw')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.0)
    if ('all' in indices) or ('gla' in indices):
        gla = np.divide((2*g-r-b), (2*g+r+b),
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        pdindex = np.vstack((pdindex, gla))
        del(gla)
        pdindexnames = np.append(pdindexnames, 'gla')
        maxarr = np.append(maxarr, -1.0)
        maxarr = np.append(maxarr, 1.0)
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
    outdat = pd.DataFrame(list(pdindex),
                          index=list(pdindexnames),
                          columns=['vals'])
    outdat['minidxpos'] = minarr   # then add a new column with the minimum possible index values
    outdat['maxidxpos'] = maxarr   # then add a new column with the maximum possible index values
    return outdat
