import argparse

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

def parse_cmd_arguments(default_vals):
    '''
    Argument parser function.

    This function takes an Args() class object with default values and populates
    default values from the command line arguments wherever they are present.
    '''
    # initialize parser
    psr = argparse.ArgumentParser()

    # add arguments
    psr.add_argument('-v','-veg','--vegfile')
    psr.add_argument('-g','-ground','--groundfile')
    psr.add_argument('-m','-name','--modelname')
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

    # parse arguments
    args = psr.parse_args()

    # create empty dictionary for arguments passed
    optionsargs = {}

    # parse command line arguments
    if args.vegfile:
        # input vegetation only dense cloud/point cloud
        default_vals.filein_vegetation = str(args.vegfile)
        optionsargs['vegetation file'] = str(args.vegfile)
    if args.groundfile:
        # input bare-Earth only dense cloud/point cloud
        default_vals.filein_ground = str(args.groundfile)
        optionsargs['bare-Earth file'] = str(args.groundfile)
    if args.modelname:
        # model output name (used to save the model)
        default_vals.model_output_name = str(args.modelname)
        optionsargs['model name'] = str(args.modelname)
    if args.vegindex:
        # because the input argument is handled as a single string, we need
        # to strip the brackets, split by the delimeter, and then re-form it
        # as a list of characters/strings
        default_vals.model_vegetation_indices = list(str(args.vegindex).split(','))
        if 'simple' in default_vals.model_vegetation_indices:
            default_vals.model_vegetation_indices = ['exr','exg','exb','exgr'] + default_vals.model_vegetation_indices
        optionsargs['vegetation indices'] = default_vals.model_vegetation_indices
    if args.modelinputs:
        # because the input argument is handled as a single string, we need
        # to strip the brackets, split by the delimeter, and then re-form it
        # as a list of characters/strings
        default_vals.model_inputs = str(args.modelinputs).split(',')
        if 'simple' in default_vals.model_inputs:
            (default_vals.model_inputs).remove('simple')
            simplelist = ['exr','exg','exb','exgr']
            for s in simplelist:
                if not s in default_vals.model_inputs:
                    default_vals.model_inputs = [s] + default_vals.model_inputs
            default_vals.model_vegetation_indices = 'simple'
        if 'all' in default_vals.model_inputs:
            (default_vals.model_inputs).remove('all')
            alllist = ['exr','exg','exb','exgr','ngrdi','mgrvi','gli','rgbvi','ikaw','gla']
            for a in alllist:
                if not a in default_vals.model_inputs:
                    default_vals.model_inputs = [a] + default_vals.model_inputs
            default_vals.model_vegetation_indices = 'all'
        optionsargs['model inputs'] = default_vals.model_inputs
    if args.modelnodes:
        # because the input argument is handled as a string, we need to
        # strip the brackets and split by the delimeter, convert each string
        # to an integer, and then re-map the converted integers to a list
        default_vals.model_nodes = list(map(int, str(args.modelnodes).split(',')))
        optionsargs['model nodes'] = default_vals.model_nodes
    if args.modeldropout:
        dval = float(args.modeldropout)
        # model dropout value must be within 0.0 and 1.0
        if 0.0 > dval and dval < 1.0:
            default_vals.model_dropout = dval
        else:
            print('Invalid dropout specified, using default probability of 0.2')
            default_vals.model_dropout = 0.2
        optionsargs['model dropout'] = default_vals.model_dropout
    if args.modelearlystop:
        # option to define early stopping criteria
        earlystopcriteria = list(str(args.modelearlystop).split(','))
        default_vals.model_early_stop_patience = int(earlystopcriteria[0])
        default_vals.model_early_stop_delta = float(earlystopcriteria[1])
        optionsargs['model early stop patience'] = default_vals.model_early_stop_patience
        optionsargs['model early stop delta'] = default_vals.model_early_stop_delta
    if args.trainingepochs:
        # define training epochs
        default_vals.training_epoch = int(args.trainingepochs)
        optionsargs['training epochs'] = default_vals.training_epoch
    if args.trainingbatchsize:
        # training batch size
        default_vals.training_batch_size = int(args.trainingbatchsize)
        optionsargs['training batch size'] = default_vals.training_batch_size
    if args.trainingcache:
        # cache data for faster training
        default_vals.training_cache = str_to_bool(args.trainingcache)
        optionsargs['training cache'] = default_vals.training_cache
    if args.trainingprefetch:
        # prefetch batches for faster training
        default_vals.training_prefetch = str_to_bool(args.trainingprefetch)
        optionsargs['training prefetch'] = default_vals.training_prefetch
    if args.trainingshuffle:
        # shuffle for training
        default_vals.training_shuffle = str_to_bool(args.trainingshuffle)
        optionsargs['training shuffle'] = default_vals.training_shuffle
    if args.trainingsplit:
        # training/validation split proportion
        default_vals.training_split = float(args.trainingsplit)
        optionsargs['training split'] = default_vals.training_split
    if args.classimbalance:
        # correct for class inbalance
        default_vals.training_class_imbalance_corr = str_to_bool(args.classimbalance)
        optionsargs['class imbalance correction'] = default_vals.training_class_imbalance_corr
    if args.datareduction:
        # reduce data volume
        default_vals.training_data_reduction = float(args.datareduction)
        optionsargs['data reduction'] = default_vals.training_data_reduction

    for k,v in optionsargs.items():
        print('{} = {}'.format(k,v))
