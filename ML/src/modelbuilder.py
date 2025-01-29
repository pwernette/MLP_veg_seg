import time

# load plotting module
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import clear_output

# load Tensorflow modules
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from IPython.display import clear_output

class PlotLearning(Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

def build_model(model_name, 
                training_tf_dataset, 
                validation_tf_dataset, 
                nodes=[8,8,8], 
                activation_fx='relu', 
                dropout_rate=0.2, 
                loss_metric='mean_squared_error', 
                model_optimizer='adam', 
                earlystopping=[], 
                dotrain=True, 
                dotrain_epochs=1000, 
                verbose=True):
    print('Building {} model...'.format(model_name))
    # the first layer should take the input features as its input
    #input_layer = input_feature_layer(model_inputs)
    print(training_tf_dataset.element_spec[0].shape[1:])
    input_layer = Input(shape=training_tf_dataset.element_spec[0].shape[1:], name='input_points')
    # l = Dense(nodes[0], activation=activation_fx)(input_feat_layer)
    if isinstance(nodes,list):
        if len(nodes)<1:
            print('No nodes specified, defaulting to 3 layers with 8 nodes each.')
            nodes = [8,8,8]
        l = Dense(nodes[0], activation=activation_fx, name=('L0_'+str(nodes[0])+'_nodes'))(input_layer)
    else:
        l = Dense(nodes, activation=activation_fx, name=('L0_'+str(nodes)+'_nodes'))(input_layer)
    # each subsequent layer (if present) should take the preceeding layer as its input
    if isinstance(nodes,list):
        if len(nodes)>1:
            for c,n in enumerate(nodes[1:]):
                l = Dense(n, activation=activation_fx, name=('L'+str(c+1)+'_'+str(n)+'_nodes'))(l)
    # add a dropout layer to reduce the chance of overfitting
    l = Dropout(dropout_rate, name='Dropout')(l)
    # flatten the output to a single Dense layer
    out = Dense(1, name='Final_Dense')(l)
    #l = Dense(2, name='Final_Dense')(l)
    # Simplify output layer to a single label
    #out = Activation('sigmoid', dtype='float32', name='Output')(l)

    # build the model
    mod = Model(inputs=input_layer, outputs=out, name=model_name)
    #mod = Model(inputs=model_inputs, outputs=out, name=model_name)
    # compile the model
    try:
        #mod.compile(loss=loss_metric,
        #              optimizer=model_optimizer)
        mod.compile(loss=loss_metric,
                      optimizer=model_optimizer,
                      metrics=['accuracy'])
    except Exception as e:
        print(e)

    # optional: print the model summary (includes structure)
    if verbose:
        print(mod.summary())

    # create history callback
    hist = History()
    if dotrain:
        call_list = [hist]
        if earlystopping:
            call_list.append(EarlyStopping(monitor='val_loss',
                                           patience=earlystopping[0],
                                           min_delta=earlystopping[1],
                                           mode='max'))
        call_list.append(ReduceLROnPlateau(monitor='val_loss', 
                                           factor=0.5, 
                                           patience=10, 
                                           min_delta=1e-4, 
                                           mode='max', 
                                           verbose=1))
        # if verbose:
        #     call_list.append(PlotLearning())
        start_time = time.time()
        mod.fit(training_tf_dataset,
                validation_data=validation_tf_dataset,
                epochs=dotrain_epochs,
                verbose=2,
                callbacks=call_list
        )
        train_time = time.time()-start_time
        print("Train time = {}s".format(train_time))

    # return the model
    return mod,hist,train_time
