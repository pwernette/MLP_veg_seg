import sys, os, math
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# import machine learning libraries
import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from tensorflow.keras.callbacks import EarlyStopping

# message box and file selection libraries
import tkinter
from tkinter import Tk
from tkinter.filedialog import askopenfile
from tkinter import simpledialog

def calculate_confusion_matrix(model, test_dataset, verbose=True):
    '''
    Calculate a confusion matrix from the test dataset specified
    '''
    # Initialize empty arrays for labels and predictions 
    model_predictions = np.array([])
    original_labels = np.array([])

    # Iterate through the training data 
    for x, y in test_dataset:
        """ Extract the training labels """
        # print('Original Labels')
        # print(original_labels)
        # print('To Concatenate:')
        # print(np.argmax(y.numpy(), axis=-1))
        original_labels = np.concatenate([original_labels, y])
        # Predict class using model 
        # model_predictions = np.concatenate([model_predictions, y])      # debugging only
        model_predictions = np.concatenate([model_predictions, tf.argmax(model.predict(x, verbose=1), axis=-1)])
        if verbose:
            print('\nOriginal:\n {}\nPredicted:\n {}'.format(y, tf.argmax(model.predict(x, verbose=1), axis=-1)))

    # Create the confusion matrix (and calculate as a percentage) 
    confusion_mat = tf.math.confusion_matrix(labels=original_labels, predictions=model_predictions).numpy()
    confusion_mat_percentage = confusion_mat/confusion_mat.sum(axis=1)[:, tf.newaxis]
    if verbose:
        print('\nConfusion matrix:\n{}'.format(confusion_mat))
    return(confusion_mat, confusion_mat_percentage)


def plot_confusion_matrices(confusion_matrices, confusion_matrices_percentage, dir, basename, class_names, verbose=True):
    plt.rcParams['figure.figsize'] = (12.0,6.5)
    plt.rcParams['figure.subplot.bottom'] = 0.3
    
    # Compute combined confusion matrix
    conf_matrix = confusion_matrices[0]
    for m in np.arange(1,len(confusion_matrices)):
        conf_matrix = [[conf_matrix[i][j] + confusion_matrices[m][i][j]  for j in range(len(conf_matrix[0]))] for i in range(len(conf_matrix))]
    
        # re-format confusion matrix as 2D array for plotting
    confusion_matrix = np.asarray(conf_matrix)

    print('\nCombined confusion matrix:')
    print(conf_mat)
    
    # Calculate the classification accuracy for each cell as a percentage
    confusion_mat_percent = conf_mat/conf_mat.sum(axis=1)[:, tf.newaxis]

    # Plot the loss training curve 
    if verbose:
        print('\nPlotting confusion matrices.')

    # Write confusion matrix to CSV file 
    f_indiv = open(os.path.join(dir, 'output_models', str(model.name)+'_confusion_matrix.csv'), 'w+', newline='')
    wr_indiv = csv.writer(f_indiv, delimiter=',')

    # Confusion matrix iterator (used for file naming)
    cmat_iter = 1

    # Plot individual model confusion matrices
    for cmat in self.confusion_matrices:
        # Plot the image count confusion matrix 
        fig = sns.heatmap(
            cmat, annot=True, fmt='d',
            xticklabels=self.cnames,
            yticklabels=self.cnames,
            cmap=sns.color_palette('gray_r')
        )
        
        # Add a title 
        fig.set_title('Image Classification')
        
        # Add X and Y axes labels 
        fig.set(xlabel='Predicted', ylabel='True')
        savefig = fig.get_figure()
        if 'eps' in self.das.plot_format:
            savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter)+'.eps'), dpi=300, format='eps')
            savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter)+'.jpg'), dpi=300)
        elif 'jpg' in self.das.plot_format or 'jpeg' in self.das.plot_format:
            savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter)+'.jpg'), dpi=300)
        elif 'tif' in self.das.plot_format:
            savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter)+'.tif'), dpi=300)
        else:
            savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter)+'.png'), dpi=300, transparent=True)

        if self.das.model_verbose_run > 0:
            print('Saved: {}'.format(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter))))
        
        # Clear plot 
        plt.clf()

        # Write results to output log file
        wr_indiv.writerow(self.cnames)
        wr_indiv.writerows(cmat)
        
        # Increment confusion matrix iterator
        cmat_iter+=1
    # Close the confusion matrix file
    f_indiv.close()
        
    # Plot combined confusion matrix 
    fig = sns.heatmap(
        self.confusion_matrix, annot=True, fmt='.3g',
        xticklabels=self.cnames,
        yticklabels=self.cnames,
        cmap=sns.color_palette('gray_r')
    )
    
    # Add a title 
    fig.set_title('Image Classification')
    
    # Add X and Y axes labels 
    fig.set(xlabel='Predicted', ylabel='True')
    savefig = fig.get_figure()
    if 'eps' in self.das.plot_format:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.eps'), dpi=300, format='eps')
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.jpg'), dpi=300)
    elif 'jpg' in self.das.plot_format:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.jpg'), dpi=300)
    elif 'tif' in self.das.plot_format:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.tif'), dpi=300)
    else:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.png'), dpi=300, transparent=True)
    if self.das.model_verbose_run > 0:
        print('Saved: {}'.format(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.png')))
    
    # Clear plot 
    plt.clf()

    # Write confusion matrix to CSV file 
    f = open(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_combined.csv'), 'w+', newline='')
    wr = csv.writer(f, delimiter=',')
    wr.writerow(self.cnames)
    wr.writerows(self.confusion_matrix)
    f.close()
    
    # Print the name of the confusion matrix file to console
    if self.das.model_verbose_run > 0:
        print('Saved: {}'.format(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_'+str(cmat_iter))))

    # Plot the percent accurate confusion matrix 
    fig = sns.heatmap(
        confusion_mat_percent, annot=True, fmt='.3g',
        xticklabels=self.cnames,
        yticklabels=self.cnames,
        cmap=sns.color_palette('gray_r')
    )
    
    # Add a title 
    fig.set_title('Image Classification (percentage)')
    
    # Add X and Y axes labels 
    fig.set(xlabel='Predicted', ylabel='True')
    savefig = fig.get_figure()
    if 'eps' in self.das.plot_format:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_percent.eps'), dpi=300, format='eps')
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_percent.jpg'), dpi=300)
    elif 'jpg' in self.das.plot_format:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_percent.jpg'), dpi=300)
    elif 'tif' in self.das.plot_format:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_percent.tif'), dpi=300)
    else:
        savefig.savefig(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_percent.png'), dpi=300, transparent=True)
    if self.das.model_verbose_run > 0:
        print('Saved: {}'.format(os.path.join(self.das.rootdir, 'output_models', str(self.mod.name)+'_confusion_matrix_percent.png')))
    
    # Clear plot 
    plt.clf()