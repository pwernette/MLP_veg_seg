# convert all input arrays to float of target type
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


def normdat(inarr, min_max=[0,65535], normrange=[0,1]):
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
    if min_max[0]!=0:
        minval = np.amin(inarr)
    if min_max[1]!=65535 or min_max[1]!=255:
        maxval = np.amax(inarr)
    # normalize the input array
    norminarr = (normrange[1]-normrange[0])*np.divide((inarr-np.asarray(minval)),
                                                    (np.asarray(maxval)-np.asarray(minval)),
                                                    out=np.zeros_like(inarr),
                                                    where=(maxval-minval)!=0)-normrange[0]
    return norminarr

# normalize RGB bands with specified color depth
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
        print("ERROR: bit-depth must be 8 or 16. Assuming 8-bit color depth")
        b1min,b1max,b2min,b2max,b3min,b3max = 0,255,0,255,0,255
    # normalize bands indivudally first
    b1norm = normdat(b1, min_max=[b1min,b1max])
    b2norm = normdat(b2, min_max=[b2min,b2max])
    b3norm = normdat(b3, min_max=[b3min,b3max])
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


'''
Otsu's thresholding function

Function is adapted from the following reference:
    Otsu, N. A threshold selection method from gray level histogram. IEEE Trans.
    Syst. Man Cybern. 1979, 9, 66–166.

The source for the code block is:
    https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python
     NOTE: The original answer does not include intensity. However, the follow-
     up answer (see the second block) DOES include intensity in the computation
     of a threshold. Since we are trying to adapt this approach to only work
     with vegetation indices, we do not need to include the recorded intensity
     when thresholding.
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


# utility for converting pd.DataFrame to tf.data
def df_to_dataset(dataframe, targetcolname='none', shuffle=True, prefetch=False, cache_ds=False, batch_size=32):
    dataframe = dataframe.copy()
    if shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    if not targetcolname == 'none':
        labels = dataframe.pop(targetcolname)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    ds = ds.batch(batch_size)
    # prefetch can significantly speed up data fetch and train time
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    # optional cache argument (for caching to memory)
    if cache_ds:
        ds = ds.cache()
    return ds

# scale XYZ coordinates for a given las/laz file
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
    # create an empty numpy array to store converted values
#     outdat = np.empty(shape=(0,int(las_file.__len__())))
    outdat = np.empty(shape=(0,len(las_file.X)))
    # SCALE X dimension
    x_dimension = las_file.X
    scale = las_file.header.scale[0]
    offset = las_file.header.offset[0]
    newrow = x_dimension*scale + offset
    outdat = np.append(outdat, np.array([newrow]), axis=0)
    # SCALE Y dimension
    y_dimension = las_file.Y
    scale = las_file.header.scale[1]
    offset = las_file.header.offset[1]
    newrow = y_dimension*scale + offset
    outdat = np.append(outdat, np.array([newrow]), axis=0)
    # SCALE Z dimension
    z_dimension = las_file.Z
    scale = las_file.header.scale[2]
    offset = las_file.header.offset[2]
    newrow = z_dimension*scale + offset
    outdat = np.append(outdat, np.array([newrow]), axis=0)
    return outdat


# function to compute one or more vegetation indices
def vegidx(lasfileobj, indices=['all'], colordepth=16):
    '''
    Compute n vegetation indicies

    RETURNS:
        1) dictionary of index names
            (useful for easily accessing the desired data from the
            output n-dimensional array)
        2) n-dimensional array with X, Y,
    '''
    '''
    Normalize all bands in a 3-band input.

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

    Returns a n-dimensional array (* denotes optional indices):

          x (:py:class:`float`)
          y (:py:class:`float`)
          z (:py:class:`float`)
          r (:py:class:`float`)
          g (:py:class:`float`)
          b (:py:class:`float`)
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

    r, g, and b values have been (presumably) normalized before
    being passed to this function. If this is not the case, then
    the code should be updated to use normalized values.

    Citations:

    Excess Red (ExR)
           Meyer, G.E.; Neto, J.C. Verification of color vegetation indices for
           automated crop imaging applications. Comput. Electron. Agric. 2008,
           63, 282–293.
    Excess Green (ExG)
           Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. Color
           Indices forWeed Identification Under Various Soil, Residue, and
           Lighting Conditions. Trans. ASAE 1995, 38, 259–269.
    Excess Blue (ExB)
           Mao,W.;Wang, Y.;Wang, Y. Real-time detection of between-row weeds
           using machine vision. In Proceedings of the 2003 ASAE Annual Meeting;
           American Society of Agricultural and Biological Engineers, Las Vegas,
           NV, USA, 27–30 July 2003.
    Excess Green minus R (ExGR)
           Neto, J.C. A combined statistical-soft computing approach for
           classification and mapping weed species in minimum -tillage systems.
           Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln, NE, USA,
           August 2004.
    Normal Green-Red Difference Index (NGRDI)
           Tucker, C.J. Red and photographic infrared linear combinations for
           monitoring vegetation. Remote Sens. Environ. 1979, 8, 127–150.
    Modified Green Red Vegetation Index (MGRVI)
           Tucker, C.J. Red and photographic infrared linear combinations for
           monitoring vegetation. Remote Sens. Environ. 1979, 8, 127–150.
    Green Leaf Index (GLI)
           Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform
           and aerial photography for documentation of grazing impacts on wheat.
           Geocarto Int. 2001, 16, 65–70.
    Red Green Blue Vegetation Index (RGBVI)
           Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.;
           Broscheit, J.; Gnyp, M.L.; Bareth, G. Combining UAV-based plant
           height from crop surface models, visible, and near infrared
           vegetation indices for biomass monitoring in barley. Int. J. Appl.
           Earth Obs. Geoinf. 2015, 39, 79–87.
    Kawashima Index (IKAW)
           Kawashima, S.; Nakatani, M. An algorithm for estimating chlorophyll
           content in leaves using a video camera. Ann. Bot. 1998, 81, 49–54.
    Green Leaf Algorithm (GLA)
           Louhaichi, M.; Borman, M.M.; Johnson, D.E. Spatially located platform
           and aerial photography for documentation of grazing impacts on wheat.
           Geocarto Int. 2001, 16, 65–70.
    '''
    # pull out the R, G, and B bands
    r = lasfileobj.red
    g = lasfileobj.green
    b = lasfileobj.blue
    # normalize R, G, and B bands
    r,g,b = normBands(r,g,b, depth=colordepth)
    # use n-dimensional numpy array as a data container
    pdindex = np.empty(shape=(0,(len(r[0]))), dtype=np.float32)
    pdindexnames = np.empty(shape=(0,0))
    # scale X, Y, and Z coordinates
    xs = scale_dims(lasfileobj)[0]
    ys = scale_dims(lasfileobj)[1]
    zs = scale_dims(lasfileobj)[2]
    # append X, Y, Z, R, G, and B (raw values) to the output n-dim numpy array
    pdindex = np.append(pdindex, [xs], axis=0)
    pdindex = np.append(pdindex, [ys], axis=0)
    pdindex = np.append(pdindex, [zs], axis=0)
    pdindex = np.append(pdindex, r, axis=0)
    pdindex = np.append(pdindex, g, axis=0)
    pdindex = np.append(pdindex, b, axis=0)
    # append index names to names array
    pdindexnames = np.append(pdindexnames,
                             ['x','y','z','r','g','b'])
    '''
    Compute vegetation indices based on user specifications

    for each index, the following steps are taken:
        1) compute the index from R, G, and/or B normalized values
        2) append the vegetation index values to the output
            multidimensional array
        3) append the vegetation index name to the output array
    '''
    if ('all' in indices) or ('exr' in indices) or ('exgr' in indices):
        exr = 1.4*b-g
        pdindex = np.append(pdindex, exr, axis=0)
        pdindexnames = np.append(pdindexnames, 'exr')
    if ('all' in indices) or ('exg' in indices) or ('exgr' in indices):
        exg = 2*g-r-b
        pdindex = np.append(pdindex, exg, axis=0)
        pdindexnames = np.append(pdindexnames, 'exg')
    if ('all' in indices) or ('exb' in indices):
        exb = 1.4*r-g
        pdindex = np.append(pdindex, exb, axis=0)
        pdindexnames = np.append(pdindexnames, 'exb')
    if ('all' in indices) or ('exgr' in indices):
        exgr = exg-exr
        pdindex = np.append(pdindex, exgr, axis=0)
        pdindexnames = np.append(pdindexnames, 'exgr')
    if ('all' in indices) or ('ngrdi' in indices):
        ngrdi = np.divide((g-r), (r+g),
                          out=np.zeros_like((g-r)),
                          where=(g+r)!=np.zeros_like(g))
        pdindex = np.append(pdindex, ngrdi, axis=0)
        pdindexnames = np.append(pdindexnames, 'ngrdi')
    if ('all' in indices) or ('mgrvi' in indices):
        mgrvi = np.divide((np.power(g,2)-np.power(r,2)),
                          (np.power(r,2)+np.power(g,2)),
                          out=np.zeros_like((g-r)),
                          where=(np.power(g,2)+np.power(r,2))!=0)
        pdindex = np.append(pdindex, mgrvi, axis=0)
        pdindexnames = np.append(pdindexnames, 'mgrvi')
    if ('all' in indices) or ('gli' in indices):
        gli = np.divide(2*g-r-b, 2*g+r+b,
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        pdindex = np.append(pdindex, gli, axis=0)
        pdindexnames = np.append(pdindexnames, 'gli')
    if ('all' in indices) or ('rgbvi' in indices):
        rgbvi = np.divide((np.power(g,2)-b*r),
                          (np.power(g,2)+b*r),
                          out=np.zeros_like((np.power(g,2)-b*r)),
                          where=(np.power(g,2)+b*r)!=0)
        pdindex = np.append(pdindex, rgbvi, axis=0)
        pdindexnames = np.append(pdindexnames, 'rgbvi')
    if ('all' in indices) or ('ikaw' in indices):
        ikaw = np.divide((r-b), (r+b),
                         out=np.zeros_like((r-b)),
                         where=(r+b)!=0)
        pdindex = np.append(pdindex, ikaw, axis=0)
        pdindexnames = np.append(pdindexnames, 'ikaw')
    if ('all' in indices) or ('gla' in indices):
        gla = np.divide((2*g-r-b), (2*g+r+b),
                        out=np.zeros_like((2*g-r-b)),
                        where=(2*g+r+b)!=0)
        pdindex = np.append(pdindex, gla, axis=0)
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
    return pdindexnames,pdindex
