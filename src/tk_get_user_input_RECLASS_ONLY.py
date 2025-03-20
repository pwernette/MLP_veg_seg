import sys
import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog

def str_to_bool(s):
    '''
    Converts str ['t','true','f','false'] to boolean, not case sensitive.
    Checks first if already a boolean.
    Raises exception if unexpected entry.
        args:
            s: str
        returns:
            out_boolean: output boolean [True or False]
    '''
    #check to see if already boolean
    if isinstance(s, bool):
        out_boolean = s
    else:
        # remove quotes, commas, and case from s
        sf = s.lower().replace('"', '').replace("'", '').replace(',', '')
        # True
        if sf in ['t', 'true']:
            out_boolean = True
        # False
        elif sf in ['f', 'false']:
            out_boolean = False
        # Unexpected arg
        else:
            # print exception so it will be visible in console, then raise exception
            print('ArgumentError: Argument invalid. Expected boolean '
                  + 'got ' + '"' + str(s) + '"' + ' instead')
            raise Exception('ArgumentError: Argument invalid. Expected boolean '
                            + 'got ' + '"' + str(s) + '"' + ' instead')
    return out_boolean


class App_reclass_only(tk.Tk):
    '''
    Create an external window and get multiple inputs from entries
    '''
    def __init__(self):
        ''' function to initialize the window '''
        super().__init__()
        # give the window a title
        self.title('Input Parameters')
    def create_widgets(self, default_arguments_obj):
        ''' function to create the window '''
        def cancel_and_exit(self):
            ''' sub-function to cancel and destroy the window '''
            self.destroy()
            sys.exit('No model name specified. Exiting program.')
        def browseFiles(intextbox, desc_text="Select a File"):
            ''' sub-function to open a file select browsing window '''
            # open a file select dialog window
            filename = filedialog.askopenfilename(title=desc_text)
            # Change textbox contents
            intextbox.delete(1.0,"end")
            intextbox.insert(1.0, filename)

        # pad x and y values for all labels, fields, and buttons in the widget
        padxval = 5
        padyval = 0

        # give the window a title
        self.title('Input Parameters')
        # iterator for row placement
        rowplacement = 0

        lab = Label(self, text='INPUT POINT CLOUD FILES:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # point cloud to reclassify
        lab = Label(self, text='Point Cloud to Reclassify')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        reclassfile = Text(self, height=1, width=50)
        reclassfile.insert(tk.END, default_arguments_obj.reclassfile)
        reclassfile.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(reclassfile, 'Select point cloud to reclassify'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        rowplacement += 1

        lab = Label(self, text='MODEL PARAMETERS:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # input model file
        lab = Label(self, text='Saved Model File')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        model_file = Text(self, height=1, width=50)
        model_file.insert(tk.END, default_arguments_obj.model_file)
        model_file.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(model_file, 'Select saved model file'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        rowplacement += 1

        def getinput(self, default_arguments_obj):
            ''' sub-function to get inputs from GUI widget '''
            # input reclassification file
            default_arguments_obj.reclassfile = reclassfile.get('1.0','end-1c').split('\n')[0]
            # input model file
            default_arguments_obj.model_file = model_file.get('1.0','end-1c').split('\n')[0]

            # after getting all values, destroy the window
            self.destroy()
        # create, define, and place submit button
        submit_button = Button(self, text="Submit", command=lambda:getinput(self, default_arguments_obj))
        submit_button.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # bind RETURN and ESC keys to sub-functions
        self.bind('<Return>', lambda event:getinput(self, default_arguments_obj))
        self.bind('<Escape>', lambda event:cancel_and_exit(self))

class Args_reclass_only():
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
        # input files:
        #   vegetation: input LAS/LAZ file containing only vegetation points
        #   ground: input LAS/LAZ file containing only bare Earth points
        #   reclassfile: input LAS/LAZ file to reclassify using the specified model
        # NOTE: If no filein_vegetation or filein_ground is/are specified, then the
        # program will default to requesting the one or both files that are missing
        # but required.
        self.filein_vegetation = 'NA'
        self.filein_ground = 'NA'
        self.reclassfile = 'NA'
        # model file used for reclassification
        self.model_file = 'NA'
        # model name:
        self.model_name = 'NA'
        
        # model inputs and vegetation indices of interest:
        #   model_inputs: list of input variables for model training and prediction
        #   model_vegetation_indices: list of vegetation indices to calculate
        #   model_nodes: list with the number of nodes per layer
        #      NOTE: The number of layers corresponds to the list length. For example:
        #          8,8,8 --> 3 layer model with 8 nodes per layer
        #          8,16 --> 2 layer model with 8 nodes (L1) and 16 nodes (L2)
        #   model_dropout: probability of dropping out (i.e. not using) any node
        #   geometry_radius: 3D radius used to compute geometry information over
        self.model_inputs = ['r','g','b']
        self.model_vegetation_indices = 'rgb'
        self.model_nodes = [8,8,8]
        self.model_dropout = 0.2
        self.geometry_radius = 0.10

        self.xyz_mins = [0,0,0]
        self.xyz_maxs = [0,0,0]
        
        # for training:
        #   epoch: number of training epochs
        #   batch size: how many records should be aggregated (i.e. batched) together
        #   prefetch: option to prefetch batches (may speed up training time)
        #   cache: option to cache batches ahead of time (speeds up training time)
        #   shuffle: option to shuffle input data (good practice)
        #   training split: proportion of the data to use for training (remainder will
        #                   be used for testing of the model performance)
        #   class imbalance corr: option to correct for class size imbalance
        #   data reduction: setting this to a number between 0 and 1 will reduce the
        #                   overall volume of data used for training and validation
        self.training_batch_size = 1000
        self.training_cache = False
        
        # for plotting:
        #   plotdir: plotting direction (horizontal (h) or vertical (v))
        self.plotdir = 'v'
    
    def parse_cmd_arguments(self):
        ''' function to update default values with any command line arguments '''
        # initialize parser
        psr = argparse.ArgumentParser()

        # add arguments
        psr.add_argument('-gui', 
                         dest='gui', 
                         type=str,
                         choices=['true','True','false','False'], 
                         default='true', 
                         help='Initialize the graphical user interface [default = true]')

        psr.add_argument('-r','-reclass','-reclassfile',
                         dest='reclassfile',
                         type=str,
                         help='Point cloud to be reclassified using the new trained model')
        psr.add_argument('-h5','-mfile','-model','-modelfile',
                         dest='modelfile',
                         type=str,
                         help='Trained MLP model file (h5 format)')
        psr.add_argument('-plotdir','-plotdir',
                         dest='plotdir',
                         type=str,
                         choices=['h','v'],
                         default='v',
                         help='(optional) Direction to orient plots [default = v (vertical)]')
        psr.add_argument('-xyzmin','-xyzmins','-xyzminimums',
                         dest='xyzmins',
                         type=float,
                         nargs=3,
                         default=[0,0,0],
                         help='(optional) Minimum values for X, Y, and Z coordinates [default = 0,0,0]')
        psr.add_argument('-xyzmax','-xyzmaxs','-xyzmaximums',
                         dest='xyzmaxs',
                         type=float,
                         nargs=3,
                         default=[0,0,0],
                         help='(optional) Maximum values for X, Y, and Z coordinates [default = 0,0,0]')
        psr.add_argument('-rad','-radius','-geometryradius',
                         dest='geometryradius',
                         type=float,
                         default=1.00,
                         help='(optional) Spherical radius over which to compute the 3D standard deviation [default = 1.00m]')

        # parse arguments
        args = psr.parse_args()

        # create empty dictionary for arguments passed
        optionsargs = {}

        # parse command line arguments
        if args.gui:
            # graphic user interface option
            self.gui = str_to_bool(args.gui)
            optionsargs['graphic user interface'] = self.gui
        if args.reclassfile:
            # input file to reclassify
            self.reclassfile = str(args.reclassfile)
            optionsargs['reclassify file'] = str(self.reclassfile)
        if args.modelfile:
            # model filename
            self.model_file = str(args.modelfile)
            optionsargs['model file'] = str(self.model_file)
        if args.plotdir:
            # plot model direction
            if args.plotdir in ['h','horizontal']:
                self.plotdir = 'LR'
            elif args.plotdir in ['v','vertical']:
                self.plotdir = 'TB'
            else:
                print('Invalid plot direction. Defaulting to vertical model plot.')
                self.plotdir = 'TB'
            optionsargs['plot direction'] = self.plotdir

        if len(optionsargs)>0:
            print('Command line parameters:')
            for k,v in optionsargs.items():
                print('  {} = {}'.format(k,v))

# # Example usage
# das = Args('das')
# das.parse_cmd_arguments()
#
# if das.gui:
#     foo = App()
#     foo.create_widgets(das)
#     foo.mainloop()
