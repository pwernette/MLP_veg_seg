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
from argclass_geom import *


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

    dat = geom_metrics(lasfileobj=dat, 
                       geom_metrics=default_values.geo_met)
    
    # get root directory
    default_values.rootdir = os.path.split(default_values.file_in)[0]
    
    default_values.file_out = os.path.join(default_values.rootdir,os.path.basename(default_values.file_in).replace(os.path.splitext(default_values.file_in)[1], '_geometry_metrics.laz'))
    print('\nUpdated output filename to: {}'.format(default_values.file_out))
    
    # write out LAZ file
    print('\nWriting new LAZ file to:\n  {}'.format(default_values.file_out))
    dat.write(default_values.file_out)

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