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

def calculate_confusion_matrix(model, test_dataset, class_depth=2, verbose=False):
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
        original_labels = np.concatenate([original_labels, np.argmax(y.numpy(), axis=-1)])
        # Predict class using model 
        # model_predictions = np.concatenate([model_predictions, y])      # debugging only
        model_predictions = np.concatenate([model_predictions, tf.argmax(model.predict(x, verbose=1), axis=-1)])
        # if verbose:
        #     print('\nOriginal:\n {}\nPredicted:\n {}'.format(y, tf.argmax(model.predict(x, verbose=1), axis=-1)))

    # Create the confusion matrix (and calculate as a percentage) 
    confusion_mat = tf.math.confusion_matrix(labels=original_labels, predictions=model_predictions).numpy()
    confusion_mat_percentage = confusion_mat/confusion_mat.sum(axis=1)[:, tf.newaxis]
    
    if verbose:
        print('\nConfusion matrix:\n{}'.format(confusion_mat))
    
    return(confusion_mat, confusion_mat_percentage)


def plot_confusion_matrices(confusion_matrices, dir, model, class_names, verbose=True):
    # set plotting parameters
    plt.rcParams['figure.figsize'] = (12.0,6.5)
    plt.rcParams['figure.subplot.bottom'] = 0.3

    # Plot the loss training curve 
    if verbose:
        print('\nPlotting confusion matrices.')

    # Write confusion matrix to CSV file 
    f_indiv = open(os.path.join(dir, str(model.name)+'_confusion_matrix.csv'), 'w+', newline='')
    wr_indiv = csv.writer(f_indiv, delimiter=',')

    # Confusion matrix iterator (used for file naming)
    cmat_iter = 1

    '''
    Plot and save individual model confusion matrices
    '''
    for cmat in confusion_matrices:
        # Plot the image count confusion matrix 
        fig = sns.heatmap(
            cmat, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap=sns.color_palette('gray_r')
        )
        
        # Add a title 
        fig.set_title('Image Classification')
        
        # Add X and Y axes labels 
        fig.set(xlabel='Predicted', ylabel='True')
        savefig = fig.get_figure()

        savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_'+str(cmat_iter)+'.eps'), dpi=300, format='eps')
        savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_'+str(cmat_iter)+'.jpg'), dpi=300)
        savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_'+str(cmat_iter)+'.png'), dpi=300, transparent=True)
        
        # Clear plot 
        plt.clf()

        # Write confusion matrix to output log file
        wr_indiv.writerow(class_names)
        wr_indiv.writerows(cmat)

        if verbose:
            print('Saved (EPS/JPG/PNG): {}'.format(os.path.join(dir, str(model.name)+'_confusion_matrix_'+str(cmat_iter))))
            print('    (results also saved to log file: {})'.format(os.path.join(dir, str(model.name)+'_confusion_matrix.csv')))

        # Increment confusion matrix iterator
        cmat_iter+=1
    # Close the confusion matrix file
    f_indiv.close()
        
    '''
    Calculate combined confusion matrix
    '''
    conf_matrix = confusion_matrices[0]
    for m in np.arange(1,len(confusion_matrices)):
        conf_matrix = [[conf_matrix[i][j] + confusion_matrices[m][i][j]  for j in range(len(conf_matrix[0]))] for i in range(len(conf_matrix))]
    
        # re-format confusion matrix as 2D array for plotting
    conf_matrix = np.asarray(conf_matrix)

    # if verbose:
    #     print('\nCombined confusion matrix:')
    #     print(conf_matrix)

    '''
    Plot combined confusion matrix
    '''
    fig = sns.heatmap(
        conf_matrix, annot=True, fmt='.3g',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap=sns.color_palette('gray_r')
    )
    
    # Add a title 
    fig.set_title('Image Classification')
    
    # Add X and Y axes labels 
    fig.set(xlabel='Predicted', ylabel='True')
    savefig = fig.get_figure()

    savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_combined.eps'), dpi=300, format='eps')
    savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_combined.jpg'), dpi=300)
    savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_combined.png'), dpi=300, transparent=True)
    
    # Clear plot 
    plt.clf()

    '''
    Write combined confusion matrix to CSV file
    '''
    f = open(os.path.join(dir, str(model.name)+'_confusion_matrix_combined.csv'), 'w+', newline='')
    wr = csv.writer(f, delimiter=',')
    wr.writerow(class_names)
    wr.writerows(conf_matrix)
    f.close()
    
    if verbose:
        print('Saved (EPS/JPG/PNG): {}'.format(os.path.join(dir, str(model.name)+'_confusion_matrix_combined.png')))
        print('    (results also saved to log file: {})'.format(os.path.join(dir, str(model.name)+'_confusion_matrix_combined.csv')))
    

    '''
    Calculate the percent accuracy confusion matrix
    '''
    confusion_mat_percent = conf_matrix/conf_matrix.sum(axis=1)[:, tf.newaxis]

    '''
    Plot the percent accurate confusion matrix
    '''
    fig = sns.heatmap(
        confusion_mat_percent, annot=True, fmt='.3f',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap=sns.color_palette('gray_r')
    )
    
    # Add a title 
    fig.set_title('Image Classification (percentage)')
    
    # Add X and Y axes labels 
    fig.set(xlabel='Predicted', ylabel='True')
    savefig = fig.get_figure()
    
    # Save the figures
    savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_percent.eps'), dpi=300, format='eps')
    savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_percent.jpg'), dpi=300)
    savefig.savefig(os.path.join(dir, str(model.name)+'_confusion_matrix_percent.png'), dpi=300, transparent=True)
    
    '''
    Write combined percent accuracy confusion matrix to CSV file
    '''
    f = open(os.path.join(dir, str(model.name)+'_confusion_matrix_percent.csv'), 'w+', newline='')
    wr = csv.writer(f, delimiter=',')
    wr.writerow(class_names)
    wr.writerows(confusion_mat_percent)
    f.close()
    
    # Print the name of the confusion matrix file to console
    if verbose:
        print('Saved: {}'.format(os.path.join(dir, str(model.name)+'_confusion_matrix_'+str(cmat_iter))))
    
    # Clear plot 
    plt.clf()


def plot_confusion_matrix(confusion_matrix, dir, model, class_names, drange='data', filename=None, plot_title=None, verbose=True):
    # set plotting parameters
    plt.rcParams['figure.figsize'] = (12.0,6.5)
    plt.rcParams['figure.subplot.bottom'] = 0.3

    # if verbose:
    #     print('\nConfusion matrix:')
    #     print(confusion_matrix)

    # switch for data-based min/max scaling or percentage-based (0.0 to 1.0)
    if drange == 'data':
        drange = [0,np.sum(confusion_matrix)]
    else:
        drange = [0.0, 1.0]
    
    # Get/Set the filename
    if filename:
        filename = filename
    else:
        filename = str(model.name)+'_confusion_matrix_'+str(drange)

    # Add a title 
    if plot_title:
        fig.set_title(plot_title)
    else:
        fig.set_title('Confusion Matrix for '+str(model.name))
    
    '''
    Plot confusion matrix
    '''
    fig = sns.heatmap(
        confusion_matrix, annot=True, fmt='.3g',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=drange[0],
        vmax=drange[1],
        cmap=sns.color_palette('gray_r')
    )
    
    

    # Add X and Y axes labels 
    fig.set(xlabel='Predicted', ylabel='True')
    savefig = fig.get_figure()

    savefig.savefig(os.path.join(dir, filename+'.eps'), dpi=300, format='eps')
    savefig.savefig(os.path.join(dir, filename+'.jpg'), dpi=300)
    savefig.savefig(os.path.join(dir, filename+'.png'), dpi=300, transparent=True)
    
    # Clear plot 
    plt.clf()

    '''
    Write combined confusion matrix to CSV file
    '''
    f = open(os.path.join(dir, str(model.name)+'_confusion_matrix.csv'), 'w+', newline='')
    wr = csv.writer(f, delimiter=',')
    wr.writerow(class_names)
    wr.writerows(confusion_matrix)
    f.close()
    
    if verbose:
        print('Saved (EPS/JPG/PNG): {}'.format(os.path.join(dir, str(model.name)+'_confusion_matrix.png')))
        print('    (results also saved to log file: {})'.format(os.path.join(dir, str(model.name)+'_confusion_matrix.csv')))
    