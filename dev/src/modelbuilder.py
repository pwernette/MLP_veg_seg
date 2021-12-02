import time

# load plotting module
import matplotlib.pyplot as plt
from IPython.display import clear_output

# load Tensorflow modules
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

class PlotLearning(tf.keras.callbacks.Callback):
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

def build_model(model_name, model_inputs, input_feature_layer, training_tf_dataset, validation_tf_dataset, nodes=[16,16,16], activation_fx='relu', dropout_rate=0.2, loss_metric='mean_squared_error', model_optimizer='adam', earlystopping=[], dotrain=True, dotrain_epochs=1000, verbose=True, plotmodel=True):
    # the first layer should take the input features as its input
    input_layer = input_feature_layer(model_inputs)
#     l = layers.Dense(nodes[0], activation=activation_fx)(input_feat_layer)
    if len(nodes)<1:
        print('No nodes specified, defaulting to 3 layers with 16 nodes each.')
        nodes = [16,16,16]
    l = layers.Dense(nodes[0], activation=activation_fx, name=(str(nodes[0])+'_nodes'))(input_layer)
    # each subsequent layer (if present) should take the preceeding layer as its input
    if len(nodes)>1:
        for n in nodes[1:]:
            l = layers.Dense(n, activation=activation_fx, name=(str(n)+'_nodes'))(l)
    # add a dropout layer to reduce the chance of overfitting
    l = layers.Dropout(dropout_rate, name='Dropout')(l)
    # flatten the output to a single Dense layer
    out = layers.Dense(1, name='Output')(l)

    # build the model
    mod = tf.keras.Model(inputs=dict(model_inputs), outputs=out, name=model_name)
    # compile the model
    try:
        mod.compile(loss=loss_metric,
                      optimizer=model_optimizer,
                      metrics=['accuracy'])
    except Exception as e:
        print(e)

    # optional: print the model summary (includes structure)
    if verbose:
        print(mod.summary)

    if dotrain:
        call_list = []
        if earlystopping:
            call_list.append(EarlyStopping(
                    monitor='accuracy',
                    patience=earlystopping[0],
                    min_delta=earlystopping[1],
                    mode='max'))
        if verbose:
            call_list.append(PlotLearning())
        start_time = time.time()
        mod.fit(training_tf_dataset,
                validation_data=validation_tf_dataset,
                epochs=dotrain_epochs,
                verbose=1,
                callbacks=call_list
        )
        print("Train time = {}s".format(time.time()-start_time))
    # plot the model as a PNG
    if plotmodel:
        plot_model(mod, to_file=('PLOT_'+model_name+'.png'), show_shapes=True, dpi=300)

    # return the model
    return mod
