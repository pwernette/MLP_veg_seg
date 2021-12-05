import numpy as np

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


def normdat(inarr, minval=0, maxval=65535, normrange=[0,1]):
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
    if minval!=0:
        minval = np.amin(inarr)
    if maxval!=65535:
        maxval = np.amax(inarr)
    norminarr = (normrange[1]-normrange[0])*np.divide((inarr-np.asarray(minval)), (np.asarray(maxval)-np.asarray(minval)), out=np.zeros_like(inarr), where=(maxval-minval)!=0)-normrange[0]
    return norminarr


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
    if depth==16 or np.amax(b1)>256 or np.amax(b2)>256 or np.amax(b3)>256:
        b1min,b1max,b2min,b2max,b3min,b3max = 0,65535,0,65535,0,65535
    elif depth==8 and np.amax(b1)<=256 and np.amax(b2)<=256 and np.amax(b3)<=256:
        b1min,b1max,b2min,b2max,b3min,b3max = 0,255,0,255,0,255
    else:
        sys.exit("ERROR: bit-depth must be 8 or 16.")
    # normalize bands indivudally first
    b1norm = normdat(b1, minval=b1min, maxval=b1max)
    b2norm = normdat(b2, minval=b2min, maxval=b2max)
    b3norm = normdat(b3, minval=b3min, maxval=b3max)
    # print(b1norm)  # FOR DEBUGGING
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
    return b1normalized, b2normalized, b3normalized
