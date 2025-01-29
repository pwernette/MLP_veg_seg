import sys, os, ntpath, laspy
import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog

from miscfx import *


class App(tk.Tk):
    '''
    Create an external window and get multiple inputs from entries
    '''
    def __init__(self):
        ''' function to initialize the window '''
        super().__init__()
        # give the window a title
        self.title('Input Parameters')
    def create_widgets(self, dao):
        ''' function to create the window '''
        def cancel_and_exit(self):
            ''' sub-function to cancel and destroy the window '''
            self.destroy()
            sys.exit('No model name specified. Exiting program.')
        def browseFiles(intextbox, desc_text="Select File"):
            ''' sub-function to open a file select browsing window '''
            # open a file select dialog window
            filenames = filedialog.askopenfilename(title=desc_text)
            # Change textbox contents
            intextbox.delete(1.0,"end")
            intextbox.insert(1.0, filenames)

        # pad x and y values for all labels, fields, and buttons in the widget
        padxval = 5
        padyval = 0

        # give the window a title
        self.title('Input Parameters')
        # iterator for row placement
        rowplacement = 0

        # training point cloud files
        lab = Label(self, text='Input Point Cloud (LAS or LAZ file):')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        file_in = Text(self, height=1, width=50)
        file_in.insert(tk.END, dao.file_in)
        file_in.grid(column=1, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # create a browse files button to get input
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(file_in, 'Select point cloud'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # vegetation indices
        lab = Label(self, text='Vegetation Index')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        veg_index = Text(self, height=1, width=50)
        veg_index.insert(tk.END, dao.veg_index)
        veg_index.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # vegetation indices
        lab = Label(self, text='Vegetation Threshold')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        veg_thresh = Text(self, height=1, width=50)
        veg_thresh.insert(tk.END, dao.veg_threshold)
        veg_thresh.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # run in verbose mode
        lab = Label(self, text='Verbose Mode (0 to 2) (def=1)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        verbose_run = Text(self, height=1, width=50)
        verbose_run.insert(tk.END, str(dao.verbose_run))
        verbose_run.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        def getinput(self, dao):
            cliargs = {}
            cliargs['python'] = 'veg_idx_comp.py'
            cliargs['-gui'] = 'False'

            ''' get inputs from GUI widget '''
            # input point cloud file
            dao.file_in = file_in.get('1.0','end-1c').split('\n')[0]
            cliargs['-pc'] = str(dao.file_in)
            
            # vegetation index
            '''['exr','exg','exb','exgr','ngrdi','mgrvi','gli','rgbvi','ikaw','gla']'''
            if " " in veg_index.get('1.0','end-1c'):
                dao.veg_index = veg_index.get('1.0','end-1c').replace(' ','_').split('\n')[0]
                dao.veg_index = [x.strip(' ') for x in dao.veg_index]
            elif not " " in veg_index.get('1.0','end-1c'):
                dao.veg_index = veg_index.get('1.0','end-1c').split('\n')[0]
            cliargs['-m'] = str(dao.veg_index)

            # input point cloud file
            dao.veg_threshold = veg_thresh.get('1.0','end-1c').split('\n')[0]
            cliargs['-t'] = float(dao.veg_threshold)
            
            # verbose mode run
            dao.verbose_run = int(verbose_run.get('1.0','end-1c').strip().split('\n')[0])
            assert dao.verbose_run >= 0, 'verbose_run must be an integer between 0 and 2'
            assert dao.verbose_run < 3, 'verbose_run must be an integer between 0 and 2'
            cliargs['-verb'] = str(dao.verbose_run)
            
            if len(cliargs)>0:
                print('CLI call:')
                for k,v in cliargs.items():
                    print('{} {} '.format(k,v))

            # after getting all values, destroy the window
            self.destroy()

        # create, define, and place submit button
        submit_button = Button(self, text="Submit", command=lambda:getinput(self, dao))
        submit_button.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # bind RETURN and ESC keys to sub-functions
        self.bind('<Return>', lambda event:getinput(self, dao))
        self.bind('<Escape>', lambda event:cancel_and_exit(self))

class Args():
    '''
    Class containing arguments
    '''
    def __init__(self, cname):
        ''' function to initialize class object with default arguments '''
        # store the name of the class object to be created
        self.name = cname

        # define default values/settings
        # to enable a graphic user interface, enable the 'gui' option
        self.gui = True

        # input file:
        #   LAS or LAZ file containing all points
        # NOTE: If no filein_vegetation or filein_ground is/are specified, then the
        # program will default to requesting the one or both files that are missing
        # but required.
        self.file_in = 'NA'
        self.rootdir = 'NA'

        # vegetation index to be calculated
        self.veg_index = 'exgr'

        # vegetation index threshold
        self.veg_threshold = 0.0

        # output filename
        self.file_out = 'NA'
        
        #   verbose run
        self.verbose_run = 2

        # laspy major version
        self.laspy_version = 0
    
    def parse_cmd_arguments(self):
        ''' function to update default values with any command line arguments '''
        # initialize parser
        psr = argparse.ArgumentParser()

        # add arguments
        psr.add_argument('-gui','-gui',
                         dest='gui',
                         type=str,
                         choices=['t','T','true','True','f','F','false','False'],
                         default='True',
                         help='Initialize program GUI [default = True]')
        psr.add_argument('-pc','-pcloud','-point_cloud',
                         dest='file_in',
                         type=list,
                         help='Training point clouds')
        psr.add_argument('-v','-vi','-index','-vindex','-veg','-vegindex',
                         dest='vegindex',
                         default='exgr',
                         type=str,
                         choices=['all','exr','exg','exb','exgr','ngrdi','mgrvi','gli','rgbvi','ikaw','gla'],
                         help='vegetation indices to use [default = all]')
        psr.add_argument('-t','-thresh','-threshold',
                         dest='veg_threshold',
                         type=float,
                         default=0.0,
                         help='Threshold value to segment point cloud from vegetation point cloud [default = 0.0]')
        psr.add_argument('-verb','-verbose_run','--verbose',
                         dest='verbose',
                         type=int,
                         default=1,
                         choices=[0,1,2],
                         help='verbose run option (0, 1, or 2)')

        # parse arguments
        args = psr.parse_args()

        # create empty dictionary for arguments passed
        optionsargs = {}

        # parse command line arguments
        # graphic user interface option
        self.gui = str_to_bool(args.gui)
        optionsargs['graphic user interface'] = self.gui

        # input vegetation only dense cloud/point cloud
        self.file_in = str(args.file_in)
        optionsargs['input file'] = str(args.file_in)
        
        if args.vegindex:
            # because the input argument is handled as a single string, we need
            # to strip the brackets, split by the delimeter, and then re-form it
            # as a list of characters/strings
            self.veg_index = str(args.vegindex).split(',')
            optionsargs['vegetation index'] = self.veg_index
        
        if not args.vegindex == 'all':
            # input vegetation only dense cloud/point cloud
            self.veg_threshold = float(args.veg_threshold)
            optionsargs['vegetation threshold'] = str(args.veg_threshold)

        if args.verbose:
            self.verbose_run = int(args.verbose)
            optionsargs['run in verbose mode'] = self.verbose_run

        if len(optionsargs)>0:
            print('Command line parameters:')
            for k,v in optionsargs.items():
                print('  {} = {}'.format(k,v))
