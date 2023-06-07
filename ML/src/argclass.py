import sys, os, ntpath
import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog

from .miscfx import *

#def getfile(window_title='Select File'):
#    '''
#    Function to open a dialog window where the user can select a single file.
#    '''
#    root_win = Tk()  # initialize the tKinter window to get user input
#    root_win.withdraw()
#    root_win.update()
#    file_io = askopenfile(title=window_title)  # get user input for the output directory
#    root_win.destroy()  # destroy the tKinter window
#    if not os.name == 'nt':  # if the OS is not Windows, then correct slashes in directory name
#        file_io = ntpath.normpath(file_io)
#    return str(file_io.name)


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
        # create, define, and place submit button
        submit_button = Button(self, text="Train Model", command=lambda:getinput(self, default_arguments_obj))
        submit_button.grid(column=4, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        rowplacement += 1

        lab = Label(self, text='Training Point Cloud Files:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # ground points training file
        lab = Label(self, text='Non-Vegetation Point Cloud')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        filein_ground = Text(self, height=1, width=50)
        filein_ground.insert(tk.END, dao.filein_ground)
        filein_ground.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # create a browse files button to get input
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(filein_ground, 'Select NON-vegetation point cloud'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # vegetation points training file
        lab = Label(self, text='Vegetation Point Cloud')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        filein_vegetation = Text(self, height=1, width=50)
        filein_vegetation.insert(tk.END, dao.filein_vegetation)
        filein_vegetation.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(filein_vegetation, 'Select vegetation point cloud'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        rowplacement += 1

        lab = Label(self, text='Reclassification Point Cloud Files:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # point cloud to reclassify
        lab = Label(self, text='Point Cloud to Reclassify')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        reclassfile = Text(self, height=1, width=50)
        reclassfile.insert(tk.END, dao.reclassfile)
        reclassfile.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(reclassfile, 'Select point cloud to reclassify'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        rowplacement += 1

        lab = Label(self, text='Model Parameters:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # saved h5 model file
        # lab = Label(self, text='Model File')
        # lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # model_file = Text(self, height=1, width=50)
        # model_file.insert(tk.END, dao.model_file)
        # model_file.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # button_explore = Button(self, text='Browse', command=lambda:browseFiles(model_file))
        # button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # rowplacement += 1

        # model name
        lab = Label(self, text='Model Name (def=<model_inputs>_<model_nodes>)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        model_output_name = Text(self, height=1, width=50)
        if dao.model_output_name == 'NA':
            model_output_name.insert(tk.END, 'model_'+dao.model_vegetation_indices+'_'+str(dao.model_nodes).replace(',','_').replace(' ','').replace('[','').replace(']',''))
        else:
            model_output_name.insert(tk.END, dao.model_output_name)
        model_output_name.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # model inputs
        lab = Label(self, text='Model Inputs (def=r,g,b)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_inputs = Text(self, height=1, width=50)
        model_inputs.insert(tk.END, str(dao.model_inputs).replace(' ','').replace('[','').replace(']',''))
        model_inputs.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # vegetation indices
        lab = Label(self, text='Vegetation Indices (def=rgb)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_vegetation_indices = Text(self, height=1, width=50)
        model_vegetation_indices.insert(tk.END, dao.model_vegetation_indices)
        model_vegetation_indices.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # model nodes
        lab = Label(self, text='Model Nodes (def=8,8,8)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_nodes = Text(self, height=1, width=50)
        model_nodes.insert(tk.END, str(dao.model_nodes).replace(' ','').replace('[','').replace(']',''))
        model_nodes.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # model dropout
        lab = Label(self, text='Model Dropout (def=0.2)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_dropout = Text(self, height=1, width=50)
        model_dropout.insert(tk.END, dao.model_dropout)
        model_dropout.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # geometry radius
        #lab = Label(self, text='Geometry Radius (opt)')
        #lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        ## get variable input
        #geometry_radius = Text(self, height=1, width=50)
        #geometry_radius.insert(tk.END, dao.geometry_radius)
        #geometry_radius.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        ## increase the row by 1
        #rowplacement += 1

        ''' training parameters '''
        lab = Label(self, text='Training Parameters:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # training epochs
        lab = Label(self, text='Training Epochs (def=100)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_epoch = Text(self, height=1, width=50)
        training_epoch.insert(tk.END, dao.training_epoch)
        training_epoch.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training batch size
        lab = Label(self, text='Training Batch Size (def=1000)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_batch_size = Text(self, height=1, width=50)
        training_batch_size.insert(tk.END, dao.training_batch_size)
        training_batch_size.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training shuffle
        lab = Label(self, text='Training Shuffle (def=True)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_shuffle = Text(self, height=1, width=50)
        training_shuffle.insert(tk.END, str(dao.training_shuffle))
        training_shuffle.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training caching
        lab = Label(self, text='Training Caching (def=True)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_cache = Text(self, height=1, width=50)
        training_cache.insert(tk.END, str(dao.training_cache))
        training_cache.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training prefetching
        lab = Label(self, text='Training Prefetching (def=True)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_prefetch = Text(self, height=1, width=50)
        training_prefetch.insert(tk.END, str(dao.training_prefetch))
        training_prefetch.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training split
        lab = Label(self, text='Validation Split (def=0.7)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_split = Text(self, height=1, width=50)
        training_split.insert(tk.END, dao.training_split)
        training_split.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # plot during training
        lab = Label(self, text='Plot Training Epochs (def=100)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_plot = Text(self, height=1, width=50)
        training_plot.insert(tk.END, str(dao.training_plot))
        training_plot.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # class imbalance correction
        lab = Label(self, text='Correct for Training Class Imbalance')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_class_imbalance_corr = Text(self, height=1, width=50)
        training_class_imbalance_corr.insert(tk.END, str(dao.training_class_imbalance_corr))
        training_class_imbalance_corr.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # data reduction
        lab = Label(self, text='Proportion of Data to Use (def=1.0)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_data_reduction = Text(self, height=1, width=50)
        training_data_reduction.insert(tk.END, dao.training_data_reduction)
        training_data_reduction.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # early stop patience
        lab = Label(self, text='Early Stop Patience (def=20')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_early_stop_patience = Text(self, height=1, width=50)
        model_early_stop_patience.insert(tk.END, dao.model_early_stop_patience)
        model_early_stop_patience.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # early stop delta
        lab = Label(self, text='Early Stop Delta (def=0.001)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        model_early_stop_delta = Text(self, height=1, width=50)
        model_early_stop_delta.insert(tk.END, dao.model_early_stop_delta)
        model_early_stop_delta.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        ''' additional parameters '''
        lab = Label(self, text='Additional Parameters:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
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

        # plot direction
        lab = Label(self, text='Plot Direction (def=v)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        plotdir = Text(self, height=1, width=50)
        plotdir.insert(tk.END, dao.plotdir)
        plotdir.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # reclassification threshold(s)
        lab = Label(self, text='Reclassification Threshold(s) (def=0.6)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        reclass_thresholds = Text(self, height=1, width=50)
        reclass_thresholds.insert(tk.END, dao.reclass_thresholds)
        reclass_thresholds.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        def getinput(self, dao):
            cliargs = {}
            cliargs['python'] = 'ML_vegfilter.py'
            cliargs['-gui'] = 'False'

            ''' sub-function to get inputs from GUI widget '''
            # input ground points file
            dao.filein_ground = filein_ground.get('1.0','end-1c').split('\n')[0]
            cliargs['-g'] = str(dao.filein_ground)
            # inputs vegetation points file
            dao.filein_vegetation = filein_vegetation.get('1.0','end-1c').split('\n')[0]
            cliargs['-v'] = str(dao.filein_vegetation)
            # input reclassification file
            dao.reclassfile = reclassfile.get('1.0','end-1c').split('\n')[0]
            cliargs['-r'] = str(dao.reclassfile)
            # input saved model file
            # dao.model_file = model_file.get('1.0','end-1c').split('\n')[0]
            # model name
            if " " in model_output_name.get('1.0','end-1c'):
                dao.model_output_name = model_output_name.get('1.0','end-1c').replace(' ','_').split('\n')[0]
                dao.model_output_name = [x.strip(' ') for x in dao.model_output_name]
            elif not " " in model_output_name.get('1.0','end-1c'):
                dao.model_output_name = model_output_name.get('1.0','end-1c').split('\n')[0]
            cliargs['-m'] = str(dao.model_output_name)
            # model inputs
            if " " in model_inputs.get('1.0','end-1c'):
                dao.model_inputs = list(model_inputs.get('1.0','end-1c').split('\n')[0].split())
                dao.model_inputs = [x.strip(' ') for x in dao.model_inputs]
            elif "," in model_inputs.get('1.0','end-1c'):
                dao.model_inputs = list(model_inputs.get('1.0','end-1c').split('\n')[0].split(','))
                dao.model_inputs = [x.replace("'",'') for x in dao.model_inputs]
                dao.model_inputs = [x.strip(' ') for x in dao.model_inputs]
            cliargs['-mi'] = str(dao.model_inputs).replace('[','').replace(']','')
            # vegetation indices
            dao.model_vegetation_indices = list(model_inputs.get('1.0','end-1c').replace("'",'').split('\n')[0].split(','))
            if 'rgb' in dao.model_vegetation_indices:
                (dao.model_vegetation_indices).remove('rgb')
                simplelist = ['r','g','b']
                for s in simplelist:
                    if not s in dao.model_inputs:
                        dao.model_inputs = [s] + dao.model_inputs
                dao.model_vegetation_indices = 'rgb'
                cliargs['-vi'] = 'rgb'
            if 'simple' in dao.model_inputs:
                (dao.model_inputs).remove('simple')
                simplelist = ['r','g','b','exr','exg','exb','exgr']
                for s in simplelist:
                    if not s in dao.model_inputs:
                        dao.model_inputs = [s] + dao.model_inputs
                dao.model_vegetation_indices = 'simple'
                cliargs['-vi'] = 'simple'
            if 'all' in dao.model_inputs:
                (dao.model_inputs).remove('all')
                alllist = ['r','g','b','exr','exg','exb','exgr','ngrdi','mgrvi','gli','rgbvi','ikaw','gla']
                for a in alllist:
                    if not a in dao.model_inputs:
                        dao.model_inputs = [a] + dao.model_inputs
                dao.model_vegetation_indices = 'all'
                cliargs['-vi'] = 'all'
            # model nodes
            if " " in model_nodes.get('1.0','end-1c'):
                dao.model_nodes = list(map(int,(model_nodes.get('1.0','end-1c').split('\n')[0].split())))
                # dao.model_nodes = [x.strip(' ') for x in dao.model_nodes]
            elif "," in model_nodes.get('1.0','end-1c'):
                dao.model_nodes = list(map(int,(model_nodes.get('1.0','end-1c').split('\n')[0].split(','))))
                # dao.model_nodes = [x.strip(' ') for x in dao.model_nodes]
            else:
                dao.model_nodes = list(map(int,(model_nodes.get('1.0','end-1c').split('\n')[0])))
            cliargs['-mn'] = str(dao.model_nodes).replace('[','').replace(']','')
            # model dropout
            if 0.0 > float(model_dropout.get('1.0','end-1c').split('\n')[0]) and float(model_dropout.get('1.0','end-1c').split('\n')[0]) < 1.0:
                dao.model_dropout = float(model_dropout.get('1.0','end-1c').replace(' ','').split('\n')[0])
            elif not 0.0 > float(model_dropout.get('1.0','end-1c').split('\n')[0]) or not float(model_dropout.get('1.0','end-1c').split('\n')[0]) < 1.0:
                dao.model_dropout = 0.2
            cliargs['-md'] = str(dao.model_dropout)
            ## geometry radius
            #dao.geometry_radius = float(geometry_radius.get('1.0','end-1c').strip().split('\n')[0])
            #cliargs['-rad'] = str(dao.geometry_radius)
            # early stop - delta
            dao.model_early_stop_delta = float(model_early_stop_delta.get('1.0','end-1c').split('\n')[0])
            # training epochs
            dao.training_epoch = int(training_epoch.get('1.0','end-1c').strip().split('\n')[0])
            assert dao.training_epoch > 0, 'number of training epochs must be greater than 0'
            cliargs['-te'] = str(dao.training_epoch)
            # early stop - patience
            dao.model_early_stop_patience = int(model_early_stop_patience.get('1.0','end-1c').split('\n')[0])
            assert dao.model_early_stop_patience <= dao.training_epoch, 'early stop patience cannot exceed the number of training epochs'
            cliargs['-mes'] = str(dao.model_early_stop_patience)+' '+str(dao.model_early_stop_delta)
            # training batch size
            dao.training_batch_size = int(training_batch_size.get('1.0','end-1c').strip().split('\n')[0])
            cliargs['-tb'] = str(dao.training_batch_size)
            # training cache
            dao.training_cache = str_to_bool(training_cache.get('1.0','end-1c').strip().split('\n')[0])
            cliargs['-tc'] = str(dao.training_cache)
            # training prefetch
            dao.training_prefetch = str_to_bool(training_prefetch.get('1.0','end-1c').strip().split('\n')[0])
            cliargs['-tp'] = str(dao.training_prefetch)
            # training split
            dao.training_split = float(training_split.get('1.0','end-1c').strip().split('\n')[0])
            assert dao.training_split > 0, 'training split must be greater than 0.0 and no more than 1.0'
            assert dao.training_split <= 1.0, 'training split must be greater than 0.0 and no more than 1.0'
            cliargs['-tsp'] = str(dao.training_split)
            # class imbalance correction
            dao.training_class_imbalance_corr = str_to_bool(training_class_imbalance_corr.get('1.0','end-1c').strip().split('\n')[0])
            cliargs['-tci'] = str(dao.training_class_imbalance_corr)
            # data reduction
            dao.training_data_reduction = float(training_data_reduction.get('1.0','end-1c').strip().split('\n')[0])
            cliargs['-tdr'] = str(dao.training_data_reduction)
            # plot training
            dao.training_plot = str_to_bool(training_plot.get('1.0','end-1c').strip().split('\n')[0])
            cliargs['-plottr'] = str(dao.training_plot)
            # plot direction
            dao.plotdir = plotdir.get('1.0','end-1c').split('\n')[0]
            cliargs['-plotdir'] = str(dao.plotdir)
            # verbose mode run
            dao.verbose_run = int(verbose_run.get('1.0','end-1c').strip().split('\n')[0])
            assert dao.verbose_run >= 0, 'verbose_run must be an integer between 0 and 2'
            assert dao.verbose_run < 3, 'verbose_run must be an integer between 0 and 2'
            cliargs['-verb'] = str(dao.verbose_run)
            # reclassification threshold(s)
            if " " in reclass_thresholds.get('1.0','end-1c'):
                dao.reclass_thresholds = list(map(float,(reclass_thresholds.get('1.0','end-1c').split('\n')[0].split())))
            elif "," in reclass_thresholds.get('1.0','end-1c'):
                dao.reclass_thresholds = list(map(float,(reclass_thresholds.get('1.0','end-1c').split('\n')[0].split(','))))
            cliargs['-thresh'] = str(dao.reclass_thresholds).replace('[','').replace(']','')
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
        # input files:
        #   vegetation: input LAS/LAZ file containing only vegetation points
        #   ground: input LAS/LAZ file containing only bare Earth points
        #   reclassfile: input LAS/LAZ file to reclassify using the specified model
        # NOTE: If no filein_vegetation or filein_ground is/are specified, then the
        # program will default to requesting the one or both files that are missing
        # but required.
        self.n_classes = 2
        self.filein_vegetation = 'NA'
        self.filein_ground = 'NA'
        self.reclassfile = 'NA'
        # model file used for reclassification
        # self.model_file = 'NA'
        # model name:
        self.model_output_name = 'NA'
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
        self.geometry_metrics = ['NA']
        self.geometry_radius = 0.10
        # model losses available for training:
        #   model_metric: metric monitored during training to determine when to stop
        #       available model metrics:
        #       ['categorical','sparse','binary','binary_crossentropy','mse','mean','mean_squared_error']
        self.model_metric = 'sparse'
        self.model_activation_function = 'relu'
        self.model_optimizer = 'adam'
        # for early stopping:
        #   delta: The minmum change required to continue training beyond the number
        #          of epochs specified by patience.
        #   patience: The number of epochs to monitor change. If there is no improvement
        #          greater than the value specified by delta, then training will stop.
        self.model_early_stop_patience = 10  # default 5
        self.model_early_stop_delta = 0.01  # default 0.001
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
        # for reclassification
        #   reclass_thresholds: list of thresholds used for reclassification
        self.reclass_thresholds = [0.6]
        # general parameters
        #   verbose run
        self.verbose_run = 2
        # laspy major version
        self.laspy_version = int(laspy.__version__.split('.')[0])
    
    def parse_cmd_arguments(self):
        ''' function to update default values with any command line arguments '''
        # initialize parser
        psr = argparse.ArgumentParser()

        # add arguments
        psr.add_argument('-gui','--gui')
        psr.add_argument('-v','-veg','--vegfile')
        psr.add_argument('-g','-ground','--groundfile')
        psr.add_argument('-r','-reclass','--reclassfile')
        # psr.add_argument('-h5','-mfile','-model','--modelfile')
        psr.add_argument('-m','-mname','--modelname')
        psr.add_argument('-vi','-index','--vegindex')
        psr.add_argument('-mi','-inputs','--modelinputs')
        psr.add_argument('-mn','-nodes','--modelnodes')
        psr.add_argument('-md','-dropout','--modeldropout')
        psr.add_argument('-ma','-activation','--modelactivation')
        psr.add_argument('-mo','-optimizer','-opt','-mopt','--modeloptimizer')
        psr.add_argument('-mes','-earlystop','--modelearlystop')
        psr.add_argument('-mm','-mmetric','-modmet','-modmetric','--modelmetric')
        psr.add_argument('-te','-epochs','--trainingepochs')
        psr.add_argument('-tb','-batch','--trainingbatchsize')
        psr.add_argument('-tc','-cache','--trainingcache')
        psr.add_argument('-tp','-prefetch','--trainingprefetch')
        psr.add_argument('-tsh','-shuffle','--trainingshuffle')
        psr.add_argument('-tsp','-split','--trainingsplit')
        psr.add_argument('-tci','-imbalance','--classimbalance')
        psr.add_argument('-tdr','-reduction','--datareduction')
        psr.add_argument('-plottr','-trainingplot','-trainplot','--plottraining')
        psr.add_argument('-plotdir','--plotdir')
        psr.add_argument('-thresh','-threshold','--reclassthresholds')
        psr.add_argument('-rad','-radius','--geometryradius')
        psr.add_argument('-verb','-verbose_run','--verbose')

        # parse arguments
        args = psr.parse_args()

        # create empty dictionary for arguments passed
        optionsargs = {}

        # parse command line arguments
        if args.gui:
            # graphic user interface option
            self.gui = str_to_bool(args.gui)
            optionsargs['graphic user interface'] = self.gui
        if args.vegfile:
            # input vegetation only dense cloud/point cloud
            self.filein_vegetation = str(args.vegfile)
            optionsargs['vegetation file'] = str(args.vegfile)
        if args.groundfile:
            # input bare-Earth only dense cloud/point cloud
            self.filein_ground = str(args.groundfile)
            optionsargs['bare-Earth file'] = str(args.groundfile)
        if args.reclassfile:
            # input file to reclassify
            self.reclassfile = str(args.reclassfile)
            optionsargs['reclassify file'] = str(self.reclassfile)
        # if args.modelfile:
        #     # model filename
        #     self.model_file = str(args.modelfile)
        #     optionsargs['model file'] = str(self.model_file)
        if args.modelname:
            # model output name (used to save the model)
            self.model_output_name = str(args.modelname)
            optionsargs['model name'] = str(args.modelname)
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
            ######### TESTING #########
            ######### TESTING #########
            if 'rgb' in self.model_inputs:
                (self.model_inputs).remove('rgb')
                simplelist = ['r','g','b']
                for s in simplelist:
                    if not s in self.model_inputs:
                        self.model_inputs = [s] + self.model_inputs
                self.model_vegetation_indices = 'rgb'
            ######### TESTING #########
            ######### TESTING #########
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
        if args.modeldropout:
            dval = float(args.modeldropout)
            # model dropout value must be within 0.0 and 1.0
            if 0.0 > dval and dval < 1.0:
                self.model_dropout = dval
            else:
                print('Invalid dropout specified, using default probability of 0.2')
                self.model_dropout = 0.2
            optionsargs['model dropout'] = self.model_dropout
        if args.modelactivation:
            # model activation function
            self.model_activation_function = str(args.modelactivation)
            optionsargs['model activation function'] = self.model_activation_function
        if args.modeloptimizer:
            # model optimizer
            self.model_optimizer = str(args.modeloptimizer)
            optionsargs['model optimizer'] = self.model_optimizer
        if args.modelmetric:
            # option to specify the metric that is monitored during training
            self.model_metric = str(args.modelmetric)
            optionsargs['model metric'] = self.model_metric
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
        if args.reclassthresholds:
            # because the input argument is handled as a single string, we need
            # to strip the brackets, split by the delimeter, and then re-form it
            # as a list of characters/strings
            self.reclass_thresholds = list(str(args.reclassthresholds).split(','))
            optionsargs['reclassification thresholds'] = self.reclass_thresholds
        if args.verbose:
            # plot model training
            self.verbose_run = int(args.verbose)
            assert self.verbose_run >= 0, 'verbose_run must be between 0 and 2'
            assert self.verbose_run < 3, 'verbose_run must be between 0 and 2'
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
