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
        if sf in ['t','true']:
            out_boolean = True
        # False
        elif sf in ['f','false']:
            out_boolean = False
        # Unexpected arg
        else:
            # print exception so it will be visible in console, then raise exception
            print('ArgumentError: Argument invalid. Expected boolean '
                  + 'got ' + '"' + str(s) + '"' + ' instead')
            raise Exception('ArgumentError: Argument invalid. Expected boolean '
                            + 'got ' + '"' + str(s) + '"' + ' instead')
    return out_boolean


class App(tk.Tk):
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
            filename = filedialog.askopenfilenames(title=desc_text)
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

        # point cloud training files
        lab = Label(self, text='Training Point Clouds')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1
        # get variable input
        filesin = Text(self, height=10, width=50)
        filesin.insert(tk.END, default_arguments_obj.filesin)
        filesin.grid(column=0, row=rowplacement, columnspan=2, rowspan=10, sticky=W, padx=padxval, pady=padyval)
        # create a browse files button to get input
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(filesin, 'Select training point clouds'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 11

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

        # saved h5 model file
        # lab = Label(self, text='Model File')
        # lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # model_file = Text(self, height=1, width=50)
        # model_file.insert(tk.END, default_arguments_obj.model_file)
        # model_file.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # button_explore = Button(self, text='Browse', command=lambda:browseFiles(model_file))
        # button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # rowplacement += 1

        # model name
        lab = Label(self, text='Model Name')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        model_output_name = Text(self, height=1, width=50)
        if default_arguments_obj.model_name == 'NA':
            model_output_name.insert(tk.END, 'model_'+str(default_arguments_obj.model_vegetation_indices).replace(' ','').replace('[','').replace(']','').replace("'","")+'_'+str(default_arguments_obj.model_nodes).replace(',','_').replace(' ','').replace('[','').replace(']',''))
        else:
            model_output_name.insert(tk.END, default_arguments_obj.model_name)
        model_output_name.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # # model inputs
        # lab = Label(self, text='Model Inputs')
        # lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # # get variable input
        # model_inputs = Text(self, height=1, width=50)
        # model_inputs.insert(tk.END, str(default_arguments_obj.model_inputs).replace(' ','').replace('[','').replace(']',''))
        # model_inputs.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # # increase the row by 1
        # rowplacement += 1

        # vegetation indices
        lab = Label(self, text='Vegetation Indices')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_vegetation_indices = Text(self, height=1, width=50)
        model_vegetation_indices.insert(tk.END, default_arguments_obj.model_vegetation_indices)
        model_vegetation_indices.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # model nodes
        lab = Label(self, text='Model Nodes')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_nodes = Text(self, height=1, width=50)
        model_nodes.insert(tk.END, str(default_arguments_obj.model_nodes).replace(' ','').replace('[','').replace(']',''))
        model_nodes.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # model dropout
        lab = Label(self, text='Model Dropout')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_dropout = Text(self, height=1, width=50)
        model_dropout.insert(tk.END, default_arguments_obj.model_dropout)
        model_dropout.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # geometry radius
        lab = Label(self, text='Geometry Radius (opt)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        geometry_radius = Text(self, height=1, width=50)
        geometry_radius.insert(tk.END, default_arguments_obj.geometry_radius)
        geometry_radius.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        ''' training parameters '''
        lab = Label(self, text='TRAINING PARAMETERS:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # training epochs
        lab = Label(self, text='Training Epochs')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_epoch = Text(self, height=1, width=50)
        training_epoch.insert(tk.END, default_arguments_obj.training_epoch)
        training_epoch.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training batch size
        lab = Label(self, text='Training Batch Size')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_batch_size = Text(self, height=1, width=50)
        training_batch_size.insert(tk.END, default_arguments_obj.training_batch_size)
        training_batch_size.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training caching
        lab = Label(self, text='Training Caching')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_cache = Text(self, height=1, width=50)
        training_cache.insert(tk.END, str(default_arguments_obj.training_cache))
        training_cache.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training prefetching
        lab = Label(self, text='Training Prefetching')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_prefetch = Text(self, height=1, width=50)
        training_prefetch.insert(tk.END, str(default_arguments_obj.training_prefetch))
        training_prefetch.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training shuffle
        lab = Label(self, text='Training Shuffle')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_shuffle = Text(self, height=1, width=50)
        training_shuffle.insert(tk.END, str(default_arguments_obj.training_shuffle))
        training_shuffle.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training split
        lab = Label(self, text='Training Split')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_split = Text(self, height=1, width=50)
        training_split.insert(tk.END, default_arguments_obj.training_split)
        training_split.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # plot during training
        lab = Label(self, text='Plot Training Epochs')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_plot = Text(self, height=1, width=50)
        training_plot.insert(tk.END, str(default_arguments_obj.training_plot))
        training_plot.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # class imbalance correction
        lab = Label(self, text='Class Imbalance Correction')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_class_imbalance_corr = Text(self, height=1, width=50)
        training_class_imbalance_corr.insert(tk.END, str(default_arguments_obj.training_class_imbalance_corr))
        training_class_imbalance_corr.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # data reduction
        lab = Label(self, text='Proportion of Data to Use')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_data_reduction = Text(self, height=1, width=50)
        training_data_reduction.insert(tk.END, default_arguments_obj.training_data_reduction)
        training_data_reduction.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # early stop patience
        lab = Label(self, text='Early Stop Patience')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_early_stop_patience = Text(self, height=1, width=50)
        model_early_stop_patience.insert(tk.END, default_arguments_obj.model_early_stop_patience)
        model_early_stop_patience.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # early stop delta
        lab = Label(self, text='Early Stop Delta')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_early_stop_delta = Text(self, height=1, width=50)
        model_early_stop_delta.insert(tk.END, default_arguments_obj.model_early_stop_delta)
        model_early_stop_delta.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        ''' additional parameters '''
        lab = Label(self, text='ADDITIONAL PARAMETERS:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # run in verbose mode
        lab = Label(self, text='Run in Verbose Mode')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        verbose_run = Text(self, height=1, width=50)
        verbose_run.insert(tk.END, str(default_arguments_obj.verbose_run))
        verbose_run.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # plot direction
        lab = Label(self, text='Plot Direction')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        plotdir = Text(self, height=1, width=50)
        plotdir.insert(tk.END, default_arguments_obj.plotdir)
        plotdir.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        def getinput(self, default_arguments_obj):
            ''' sub-function to get inputs from GUI widget '''
            # input point cloud files
            print(filesin.get('1.0','end-1c'))
            if " " in filesin.get('1.0','end-1c'):
                default_arguments_obj.filesin = list(filesin.get('1.0','end-1c').split('\n')[0].split(' '))
                default_arguments_obj.filesin = [r'{}'.format(x.strip(' ')) for x in default_arguments_obj.filesin]
            if "," in filesin.get('1.0','end-1c'):
                default_arguments_obj.filesin = list(filesin.get('1.0','end-1c').split('\n')[0].split(','))
                default_arguments_obj.filesin = [x.replace("'",'') for x in default_arguments_obj.filesin]
                default_arguments_obj.filesin = [r'{}'.format(x.strip(' ')) for x in default_arguments_obj.filesin]
            elif ";" in filesin.get('1.0','end-1c'):
                default_arguments_obj.filesin = list(filesin.get('1.0','end-1c').split('\n')[0].split(';'))
                default_arguments_obj.filesin = [x.replace("'",'') for x in default_arguments_obj.filesin]
                default_arguments_obj.filesin = [x.strip(' ') for x in default_arguments_obj.filesin]
            print(default_arguments_obj.filesin)
            # input reclassification file
            default_arguments_obj.reclassfile = reclassfile.get('1.0','end-1c').split('\n')[0]
            # input saved model file
            # default_arguments_obj.model_file = model_file.get('1.0','end-1c').split('\n')[0]
            # model name
            if " " in model_output_name.get('1.0','end-1c'):
                default_arguments_obj.model_name = model_output_name.get('1.0','end-1c').replace(' ','_').split('\n')[0]
                default_arguments_obj.model_name = [x.strip(' ') for x in default_arguments_obj.model_output_name]
            elif not " " in model_output_name.get('1.0','end-1c'):
                default_arguments_obj.model_name = model_output_name.get('1.0','end-1c').split('\n')[0]
            # # model inputs
            # if " " in model_inputs.get('1.0','end-1c'):
            #     default_arguments_obj.model_inputs = list(model_inputs.get('1.0','end-1c').split('\n')[0].split())
            #     default_arguments_obj.model_inputs = [x.strip(' ') for x in default_arguments_obj.model_inputs]
            # elif "," in model_inputs.get('1.0','end-1c'):
            #     default_arguments_obj.model_inputs = list(model_inputs.get('1.0','end-1c').split('\n')[0].split(','))
            #     default_arguments_obj.model_inputs = [x.replace("'",'') for x in default_arguments_obj.model_inputs]
            #     default_arguments_obj.model_inputs = [x.strip(' ') for x in default_arguments_obj.model_inputs]
            
            # vegetation indices
            default_arguments_obj.model_vegetation_indices = list(model_vegetation_indices.get('1.0','end-1c').replace("'",'').split('\n')[0].split(','))
            if 'rgb' in default_arguments_obj.model_vegetation_indices:
                (default_arguments_obj.model_vegetation_indices).remove('rgb')
                simplelist = ['r','g','b']
                for s in simplelist:
                    if not s in default_arguments_obj.model_inputs:
                        default_arguments_obj.model_inputs = [s] + default_arguments_obj.model_inputs
                default_arguments_obj.model_vegetation_indices = 'rgb'
            if 'simple' in default_arguments_obj.model_inputs:
                (default_arguments_obj.model_inputs).remove('simple')
                simplelist = ['r','g','b','exr','exg','exb','exgr']
                for s in simplelist:
                    if not s in default_arguments_obj.model_inputs:
                        default_arguments_obj.model_inputs = [s] + default_arguments_obj.model_inputs
                default_arguments_obj.model_vegetation_indices = 'simple'
            if 'all' in default_arguments_obj.model_inputs:
                (default_arguments_obj.model_inputs).remove('all')
                alllist = ['r','g','b','exr','exg','exb','exgr','ngrdi','mgrvi','gli','rgbvi','ikaw','gla']
                for a in alllist:
                    if not a in default_arguments_obj.model_inputs:
                        default_arguments_obj.model_inputs = [a] + default_arguments_obj.model_inputs
                default_arguments_obj.model_vegetation_indices = 'all'

            # model nodes
            if " " in model_nodes.get('1.0','end-1c'):
                default_arguments_obj.model_nodes = list(map(int,(model_nodes.get('1.0','end-1c').split('\n')[0].split())))
                # default_arguments_obj.model_nodes = [x.strip(' ') for x in default_arguments_obj.model_nodes]
            elif "," in model_nodes.get('1.0','end-1c'):
                default_arguments_obj.model_nodes = list(map(int,(model_nodes.get('1.0','end-1c').split('\n')[0].split(','))))
                # default_arguments_obj.model_nodes = [x.strip(' ') for x in default_arguments_obj.model_nodes]
            # model dropout
            if 0.0 > float(model_dropout.get('1.0','end-1c').split('\n')[0]) and float(model_dropout.get('1.0','end-1c').split('\n')[0]) < 1.0:
                default_arguments_obj.model_dropout = float(model_dropout.get('1.0','end-1c').replace(' ','').split('\n')[0])
            elif not 0.0 > float(model_dropout.get('1.0','end-1c').split('\n')[0]) or not float(model_dropout.get('1.0','end-1c').split('\n')[0]) < 1.0:
                default_arguments_obj.model_dropout = 0.2
            # geometry radius
            default_arguments_obj.geometry_radius = float(geometry_radius.get('1.0','end-1c').strip().split('\n')[0])
            # early stop - patience
            default_arguments_obj.model_early_stop_patience = int(model_early_stop_patience.get('1.0','end-1c').split('\n')[0])
            # early stop - delta
            default_arguments_obj.model_early_stop_delta = float(model_early_stop_delta.get('1.0','end-1c').split('\n')[0])
            # training epochs
            default_arguments_obj.training_epoch = int(training_epoch.get('1.0','end-1c').strip().split('\n')[0])
            # training batch size
            default_arguments_obj.training_batch_size = int(training_batch_size.get('1.0','end-1c').strip().split('\n')[0])
            # training cache
            default_arguments_obj.training_cache = str_to_bool(training_cache.get('1.0','end-1c').strip().split('\n')[0])
            # training prefetch
            default_arguments_obj.training_prefetch = str_to_bool(training_prefetch.get('1.0','end-1c').strip().split('\n')[0])
            # training split
            default_arguments_obj.training_split = float(training_split.get('1.0','end-1c').strip().split('\n')[0])
            # class imbalance correction
            default_arguments_obj.training_class_imbalance_corr = str_to_bool(training_class_imbalance_corr.get('1.0','end-1c').strip().split('\n')[0])
            # data reduction
            default_arguments_obj.training_data_reduction = float(training_data_reduction.get('1.0','end-1c').strip().split('\n')[0])
            # plot training
            default_arguments_obj.training_plot = str_to_bool(training_plot.get('1.0','end-1c').strip().split('\n')[0])
            # plot direction
            default_arguments_obj.plotdir = plotdir.get('1.0','end-1c').split('\n')[0]
            # verbose mode run
            default_arguments_obj.verbose_run = str_to_bool(verbose_run.get('1.0','end-1c').strip().split('\n')[0])
            # after getting all values, destroy the window
            self.destroy()
        # create, define, and place submit button
        submit_button = Button(self, text="Submit", command=lambda:getinput(self, default_arguments_obj))
        submit_button.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # bind RETURN and ESC keys to sub-functions
        self.bind('<Return>', lambda event:getinput(self, default_arguments_obj))
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
        # input files:
        #   filesin: input LAS/LAZ files containing points per each class
        #   reclassfile: input LAS/LAZ file to reclassify using the specified model
        # NOTE: If no filein_vegetation or filein_ground is/are specified, then the
        # program will default to requesting the one or both files that are missing
        # but required.
        self.filesin = []
        self.reclassfile = 'NA'
        # model file used for reclassification
        # self.model_file = 'NA'
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
        self.model_inputs = []
        self.model_vegetation_indices = 'rgb'
        self.model_nodes = [8,8,8]
        self.model_dropout = 0.2
        self.geometry_radius = 0.10
        self.xyz_mins = [0,0,0]
        self.xyz_maxs = [0,0,0]
        # for early stopping:
        #   delta: The minmum change required to continue training beyond the number
        #          of epochs specified by patience.
        #   patience: The number of epochs to monitor change. If there is no improvement
        #          greater than the value specified by delta, then training will stop.
        self.model_early_stop_patience = 15
        self.model_early_stop_delta = 0.001
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
        self.training_epoch = 100
        self.training_batch_size = 1000
        self.training_cache = True
        self.training_prefetch = True
        self.training_shuffle = True
        self.training_split = 0.7
        self.training_class_imbalance_corr = True
        self.training_data_reduction = 1.0
        # for plotting:
        #   plotdir: plotting direction (horizontal (h) or vertical (v))
        self.plotdir = 'v'
        self.training_plot = True
        # general parameters
        #   verbose run
        self.verbose_run = True
    
    def parse_cmd_arguments(self):
        ''' function to update default values with any command line arguments '''
        # initialize parser
        psr = argparse.ArgumentParser()

        # add arguments
        psr.add_argument('-gui', 
                         dest='gui', 
                         type=str,
                         choices=['true','True','false','False','t','f'], 
                         default='true', 
                         help='Initialize the graphical user interface [default = true]')
        psr.add_argument('-pcs','-pclouds','-pointclouds',
                         dest='filesin',
                         type=str,
                         help='Training point clouds separated by class')
        psr.add_argument('-r','-reclass','-reclassfile',
                         dest='reclassfile',
                         type=str,
                         help='Point cloud to be reclassified using the new trained model')
        psr.add_argument('-m','-mname','-modelname',
                         dest='modelname',
                         type=str,
                         default='NA',
                         help='(optional) Specify the output model file name')
        psr.add_argument('-v','-vi','-index','-vegindex',
                         dest='vegindex',
                         type=str,
                         default='rgb',
                         help='(optional) Which vegetation indices should be included in the model [default = rgb]')
        psr.add_argument('-mi','-inputs','-modelinputs',
                         dest='modelinputs',
                         type=str,
                         help='(optional) What are the model inputs')
        psr.add_argument('-mn','-nodes','-modelnodes',
                         dest='modelnodes',
                         type=str,
                         default='16,16,16',
                         help='(optional) List of integers representing the number of nodes for each layer [default = 16,16,16]')
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
        psr.add_argument('-md','-dropout','-modeldropout',
                         dest='modeldropout',
                         type=float,
                         default=0.2,
                         help='(optional) Probabilty of node dropout in the model [default = 0.2]')
        psr.add_argument('-mes','-earlystop','-modelearlystop',
                         dest='modelearlystop',
                         help='(optional) Early stopping criteria (can reduce training time and minimize overfitting)')
        psr.add_argument('-te','-epochs','-trainingepochs',
                         dest='trainingepochs',
                         type=int,
                         default=100,
                         help='(optional) Number of epochs to train the model [default = 100]')
        psr.add_argument('-tb','-batch','-trainingbatchsize',
                         dest='trainingbatchsize',
                         type=int,
                         default=1000,
                         help='(optional) Batch size to use during model training [default = 1000]')
        psr.add_argument('-tc','-cache','-trainingcache',
                         dest='trainingcache',
                         type=str,
                         choices=['true','True','false','False','t','f'],
                         default='false',
                         help='(optional) Cache data in RAM to reduce training time (WARNING: May run into OOM errors) [deafult = false]')
        psr.add_argument('-tp','-prefetch','-trainingprefetch',
                         dest='trainingprefetch',
                         type=str,
                         choices=['true','True','false','False','t','f'],
                         default='true',
                         help='(optional) Prefetch data during training to reduce training time [default = true]')
        psr.add_argument('-tsh','-shuffle','-trainingshuffle',
                         dest='trainingshuffle',
                         type=str,
                         choices=['true','True','false','False','t','f'],
                         default='true',
                         help='(optional) Shuffle training data (HIGHLY RECOMMENDED) [default = true]')
        psr.add_argument('-tsp','-split','-trainingsplit',
                         dest='trainingsplit',
                         type=float,
                         default=0.7,
                         help='(optional) Training split (i.e., proportion of the data used for training the model) [default = 0.7]')
        psr.add_argument('-tci','-imbalance','-classimbalance',
                         dest='classimbalance',
                         type=str,
                         choices=['true','True','false','False','t','f'],
                         default='true',
                         help='(optional) Undersample minority class to equalize class representation [default = true]')
        psr.add_argument('-tdr','-reduction','-datareduction',
                         dest='datareduction',
                         type=float,
                         default=1.0,
                         help='(optional) Use this proportion of the data total [default = 1.0]')
        psr.add_argument('-rad','-radius','-geometryradius',
                         dest='geometryradius',
                         type=float,
                         default=1.00,
                         help='(optional) Spherical radius over which to compute the 3D standard deviation [default = 1.00m]')
        psr.add_argument('-ptrain','-plot','-plot_train','-plottrain','-plottr','-trainingplot','-trainplot','-plottraining',
                         dest='plottraining',
                         type=str,
                         choices=['true','True','false','False','t','f'],
                         default='true',
                         help='(optional) Plot training history [default = true]')
        psr.add_argument('-plotdir','-plotdir',
                         dest='plotdir',
                         type=str,
                         choices=['h','horizontal','v','vertical'],
                         default='v',
                         help='(optional) Direction to orient plots [default = v (vertical)]')
        psr.add_argument('-verb','-verbose_run','-verbose',
                         dest='verbose',
                         type=str,
                         choices=['true','True','false','False','t','f'],
                         help='(optional) Run model in verbose mode (will print more information to the console and run slower) [default = false]')

        # parse arguments
        args = psr.parse_args()

        # create empty dictionary for arguments passed
        optionsargs = {}

        # parse command line arguments
        if args.gui:
            # graphic user interface option
            self.gui = str_to_bool(args.gui)
            optionsargs['graphic user interface'] = self.gui
        if args.filesin:
            # input vegetation only dense cloud/point cloud
            self.filesin = list(map(str,str(args.filesin).replace(' ','').split(',')))
            optionsargs['training point cloud files'] = str(args.filesin)
        if args.reclassfile:
            # input file to reclassify
            self.reclassfile = str(args.reclassfile)
            optionsargs['reclassify file'] = str(self.reclassfile)
        # if args.modelfile:
        #     # model filename
        #     self.model_file = str(args.modelfile)
        #     optionsargs['model file'] = str(self.model_file)
        if args.vegindex:
            # because the input argument is handled as a single string, we need
            # to strip the brackets, split by the delimeter, and then re-form it
            # as a list of characters/strings
            self.model_vegetation_indices = list(str(args.vegindex).split(','))
            if 'simple' in self.model_vegetation_indices:
                self.model_vegetation_indices = ['exr','exg','exb','exgr'] + self.model_vegetation_indices
            optionsargs['vegetation indices'] = self.model_vegetation_indices
        if args.modelinputs:
            # because the input argument is handled as a single string, we need
            # to strip the brackets, split by the delimeter, and then re-form it
            # as a list of characters/strings
            self.model_inputs = list(str(args.modelinputs).split(','))
            if 'rgb' in self.model_inputs:
                (self.model_inputs).remove('rgb')
                simplelist = ['r','g','b']
                for s in simplelist:
                    if not s in self.model_inputs:
                        self.model_inputs = [s] + self.model_inputs
                self.model_vegetation_indices = 'rgb'
            if 'simple' in self.model_inputs:
                (self.model_inputs).remove('simple')
                simplelist = ['exr','exg','exb','exgr']
                for s in simplelist:
                    if not s in self.model_inputs:
                        self.model_inputs = [s] + self.model_inputs
                self.model_vegetation_indices = 'simple'
            if 'all' in self.model_inputs:
                (self.model_inputs).remove('all')
                alllist = ['exr','exg','exb','exgr','ngrdi','mgrvi','gli','rgbvi','ikaw','gla']
                for a in alllist:
                    if not a in self.model_inputs:
                        self.model_inputs = [a] + self.model_inputs
                self.model_vegetation_indices = 'all'
            optionsargs['model inputs'] = self.model_inputs
        if args.modelnodes:
            # because the input argument is handled as a string, we need to
            # strip the brackets and split by the delimeter, convert each string
            # to an integer, and then re-map the converted integers to a list
            self.model_nodes = list(map(int, str(args.modelnodes).split(',')))
            optionsargs['model nodes'] = self.model_nodes
        if args.modelname:
            # model output name (used to save the model)
            if args.modelname == 'NA':
                self.model_name = 'model_'+str(args.vegindex)+'_'+str(args.modelnodes).replace(',','_').replace(' ','').replace('[','').replace(']','')
            else:
                self.model_name = str(args.modelname)
            # replace commas with underscores
            self.model_name = self.model_name.replace(',','_')
            optionsargs['model name'] = str(args.modelname)
        if args.modeldropout:
            dval = float(args.modeldropout)
            # model dropout value must be within 0.0 and 1.0
            if 0.0 > dval and dval < 1.0:
                self.model_dropout = dval
            else:
                print('Invalid dropout specified, using default probability of 0.2')
                self.model_dropout = 0.2
            optionsargs['model dropout'] = self.model_dropout
        if args.geometryradius:
            # option to specify geometry radius will only be used IF one of the
            # geometry metrics is specified
            self.geometry_radius = float(args.geometryradius)
            optionsargs['geometry radius'] = self.geometry_radius
        if args.modelearlystop:
            # option to define early stopping criteria
            earlystopcriteria = list(str(args.modelearlystop).split(','))
            self.model_early_stop_patience = int(earlystopcriteria[0])
            self.model_early_stop_delta = float(earlystopcriteria[1])
            optionsargs['model early stop patience'] = self.model_early_stop_patience
            optionsargs['model early stop delta'] = self.model_early_stop_delta
        if args.trainingepochs:
            # define training epochs
            self.training_epoch = int(args.trainingepochs)
            optionsargs['training epochs'] = self.training_epoch
        if args.trainingbatchsize:
            # training batch size
            self.training_batch_size = int(args.trainingbatchsize)
            optionsargs['training batch size'] = self.training_batch_size
        if args.trainingcache:
            # cache data for faster training
            self.training_cache = str_to_bool(args.trainingcache)
            optionsargs['training cache'] = self.training_cache
        if args.trainingprefetch:
            # prefetch batches for faster training
            self.training_prefetch = str_to_bool(args.trainingprefetch)
            optionsargs['training prefetch'] = self.training_prefetch
        if args.trainingshuffle:
            # shuffle for training
            self.training_shuffle = str_to_bool(args.trainingshuffle)
            optionsargs['training shuffle'] = self.training_shuffle
        if args.trainingsplit:
            # training/validation split proportion
            self.training_split = float(args.trainingsplit)
            optionsargs['training split'] = self.training_split
        if args.classimbalance:
            # correct for class inbalance
            self.training_class_imbalance_corr = str_to_bool(args.classimbalance)
            optionsargs['class imbalance correction'] = self.training_class_imbalance_corr
        if args.datareduction:
            # reduce data volume
            self.training_data_reduction = float(args.datareduction)
            optionsargs['data reduction'] = self.training_data_reduction
        if args.plottraining:
            # plot model training
            self.training_plot = str_to_bool(args.plottraining)
            optionsargs['plot during training'] = self.training_plot
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
        if args.verbose:
            # plot model training
            self.verbose_run = str_to_bool(args.verbose)
            optionsargs['run in verbose mode'] = self.verbose_run

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
