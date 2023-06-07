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
        lab = Label(self, text='Saved h5 Model File')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        model_file = Text(self, height=1, width=50)
        model_file.insert(tk.END, default_arguments_obj.model_file)
        model_file.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        button_explore = Button(self, text='Browse', command=lambda:browseFiles(model_file, 'Select saved h5 model file'))
        button_explore.grid(column=3, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        rowplacement += 1

        ''' predicting parameters '''
        lab = Label(self, text='RECLASSIFICATION PARAMETERS:')
        lab.grid(column=0, columnspan=3, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        rowplacement += 1

        # reclassification threshold(s)
        lab = Label(self, text='Reclassification Threshold(s)')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        reclass_thresholds = Text(self, height=1, width=50)
        reclass_thresholds.insert(tk.END, default_arguments_obj.reclass_thresholds)
        reclass_thresholds.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training batch size
        lab = Label(self, text='Batch Size')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_batch_size = Text(self, height=1, width=50)
        training_batch_size.insert(tk.END, default_arguments_obj.training_batch_size)
        training_batch_size.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training caching
        lab = Label(self, text='Caching')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_cache = Text(self, height=1, width=50)
        training_cache.insert(tk.END, str(default_arguments_obj.training_cache))
        training_cache.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        # training prefetching
        lab = Label(self, text='Prefetching')
        lab.grid(column=0, row=rowplacement, sticky=W, padx=padxval, pady=padyval)
        # get variable input
        training_prefetch = Text(self, height=1, width=50)
        training_prefetch.insert(tk.END, str(default_arguments_obj.training_prefetch))
        training_prefetch.grid(column=1, row=rowplacement, sticky=E, padx=padxval, pady=padyval)
        # increase the row by 1
        rowplacement += 1

        def getinput(self, default_arguments_obj):
            ''' sub-function to get inputs from GUI widget '''
            # input reclassification file
            default_arguments_obj.reclassfile = reclassfile.get('1.0','end-1c').split('\n')[0]
            # input model file
            default_arguments_obj.model_file = model_file.get('1.0','end-1c').split('\n')[0]
            # batch size
            default_arguments_obj.training_batch_size = int(training_batch_size.get('1.0','end-1c').strip().split('\n')[0])
            # cache
            default_arguments_obj.training_cache = str_to_bool(training_cache.get('1.0','end-1c').strip().split('\n')[0])
            # prefetch
            default_arguments_obj.training_prefetch = str_to_bool(training_prefetch.get('1.0','end-1c').strip().split('\n')[0])
            # reclassification threshold(s)
            if " " in reclass_thresholds.get('1.0','end-1c'):
                default_arguments_obj.reclass_thresholds = list(map(float,(reclass_thresholds.get('1.0','end-1c').split('\n')[0].split())))
            elif "," in reclass_thresholds.get('1.0','end-1c'):
                default_arguments_obj.reclass_thresholds = list(map(float,(reclass_thresholds.get('1.0','end-1c').split('\n')[0].split(','))))
            elif ";" in reclass_thresholds.get('1.0','end-1c'):
                default_arguments_obj.reclass_thresholds = list(map(float,(reclass_thresholds.get('1.0','end-1c').split('\n')[0].split(';'))))
            else:
                default_arguments_obj.reclass_thresholds = float(reclass_thresholds.get('1.0','get-1c').split('\n')[0])
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
        # for early stopping:
        #   delta: The minmum change required to continue training beyond the number
        #          of epochs specified by patience.
        #   patience: The number of epochs to monitor change. If there is no improvement
        #          greater than the value specified by delta, then training will stop.
        self.model_early_stop_patience = 5
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
        # for reclassification
        #   reclass_thresholds: list of thresholds used for reclassification
        self.reclass_thresholds = [0.6]
    def parse_cmd_arguments(self):
        ''' function to update default values with any command line arguments '''
        # initialize parser
        psr = argparse.ArgumentParser()

        # add arguments
        psr.add_argument('-gui','--gui')
        psr.add_argument('-v','-veg','--vegfile')
        psr.add_argument('-g','-ground','--groundfile')
        psr.add_argument('-r','-reclass','--reclassfile')
        psr.add_argument('-h5','-mfile','-model','--modelfile')
        psr.add_argument('-m','-mname','--modelname')
        psr.add_argument('-vi','-index','--vegindex')
        psr.add_argument('-mi','-inputs','--modelinputs')
        psr.add_argument('-mn','-nodes','--modelnodes')
        psr.add_argument('-md','-dropout','--modeldropout')
        psr.add_argument('-mes','-earlystop','--modelearlystop')
        psr.add_argument('-te','-epochs','--trainingepochs')
        psr.add_argument('-tb','-batch','--trainingbatchsize')
        psr.add_argument('-tc','-cache','--trainingcache')
        psr.add_argument('-tp','-prefetch','--trainingprefetch')
        psr.add_argument('-tsh','-shuffle','--trainingshuffle')
        psr.add_argument('-tsp','-split','--trainingsplit')
        psr.add_argument('-tci','-imbalance','--classimbalance')
        psr.add_argument('-tdr','-reduction','--datareduction')
        psr.add_argument('-plotdir','--plotdir')
        psr.add_argument('-thresh','-threshold','--reclassthresholds')
        psr.add_argument('-rad','-radius','--geometryradius')

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
        if args.modelfile:
            # model filename
            self.model_file = str(args.modelfile)
            optionsargs['model file'] = str(self.model_file)
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
