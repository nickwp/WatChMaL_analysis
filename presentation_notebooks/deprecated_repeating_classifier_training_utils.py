# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os,sys
import pandas as pd
import numpy as np
import math
import itertools
from functools import reduce
import h5py

from progressbar import *

from sklearn.metrics import roc_curve, auc
from sklearn.utils.extmath import stable_cumsum

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from seaborn import heatmap

from textwrap import wrap

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def disp_learn_hist_smoothed(location, losslim=None, window_train=400,window_val=40,show=True, leg_size=24):
    train_log=location+'/log_train.csv'
    val_log=location+'/log_val.csv'
    
    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)

    epoch_train    = moving_average(np.array(train_log_csv.epoch),window_train)
    accuracy_train = moving_average(np.array(train_log_csv.accuracy),window_train)
    loss_train     = moving_average(np.array(train_log_csv.loss),window_train)
    
    epoch_val    = moving_average(np.array(val_log_csv.epoch),window_val)
    accuracy_val = moving_average(np.array(val_log_csv.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_csv.loss),window_val)

    epoch_val_uns    = np.array(val_log_csv.epoch)
    accuracy_val_uns = np.array(val_log_csv.accuracy)
    loss_val_uns     = np.array(val_log_csv.loss)

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(epoch_train, loss_train, linewidth=2, label='Average training loss', color='b', alpha=0.3)
    line12 = ax1.plot(epoch_val, loss_val, label='Average validation loss', color='blue')

    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(epoch_train, accuracy_train, linewidth=2, label='Average training accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(epoch_val, accuracy_val, label='Average validation accuracy', color='red')    
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)

    lines  = line11+ line12+ line21+ line22

    labels = [l.get_label() for l in lines]
    
    leg    = ax2.legend(lines, labels,  loc=5, numpoints=1,prop={'size' : leg_size})
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return

    return fig

def plot_confusion_matrix(labels, predictions, class_names,title=None):
    """
    plot_confusion_matrix(labels, predictions, class_names)
    
    Plots the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
          title               ... optional, title for the confusion matrix
    """
    
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(predictions, labels,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

    # Normalize the confusion matrix
    mat = mat.astype("float") / mat.sum(axis=0)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20) 
        
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=20)
    ax.set_yticklabels(class_names,fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    if title is not None: 
        ax.set_title(title,fontsize=20)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                    ha="center", va="center", fontsize=20,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    plt.show()

def plot_multiple_ROC(data, metric, pos_neg_labels, plot_labels = None, png_name=None,title='ROC Curve', annotate=True,ax=None, linestyle=None, leg_loc=None, xlabel=None,ylabel=None,legend_label_dict=None):
    '''
    plot_multiple_ROC(data, metric, pos_neg_labels, plot_labels = None, png_name=None,title='ROC Curve', annotate=True,ax=None, linestyle=None, leg_loc=None, xlabel=None,ylabel=None,legend_label_dict=None)

    Plot multiple ROC curves of background rejection vs signal efficiency. Can plot 'rejection' (1/fpr) or 'fraction' (tpr).
    Args:
        data                ... tuple of (n false positive rates, n true positive rate, n thresholds) to plot rejection or 
                                (rejection fractions, true positive rates, false positive rates, thresholds) to plot rejection fraction.
        metric              ... string, name of metric to plot: ('rejection' or 'fraction')
        pos_neg_labels      ... array of one positive and one negative string label, or list of lists, with each list giving positive and negative label for
                                one dataset
        plot_labels         ... label for each run to display in legend
        png_name            ... name of image to save
        title               ... title of plot
        annotate            ... whether or not to include annotations of critical points for each curve, default True
        ax                  ... matplotlib.pyplot.axes on which to place plot
        linestyle           ... list of linestyles to use for each curve, can be '-', ':', '-.'
        leg_loc             ... location for legend, eg 'upper right' - vertical upper, center, lower, horizontal right left
        legend_label_dict   ... dictionary of display symbols for each string label, to use for displaying pretty characters in labels
    author: Calum Macdonald
    June 2020
    '''

    if legend_label_dict is None:
        legend_label_dict={}
        if isinstance(pos_neg_labels[0], str):
            legend_label_dict[pos_neg_labels[0]]=pos_neg_labels[0]
            legend_label_dict[pos_neg_labels[1]]=pos_neg_labels[1]
        else:
            for j in range(len(pos_neg_labels)):
                legend_label_dict[pos_neg_labels[j][0]]=pos_neg_labels[j][0]
                legend_label_dict[pos_neg_labels[j][1]]=pos_neg_labels[j][1]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16,9),facecolor="w")
        ax.tick_params(axis="both", labelsize=20)
    
    model_colors = [np.random.rand(3,) for i in data[0]]
    
    for j in np.arange(len(data[0])):
        if isinstance(pos_neg_labels[0], str):
            label_0 = pos_neg_labels[0]
            label_1 = pos_neg_labels[1]
        else:
            label_0 = pos_neg_labels[j][0]
            label_1 = pos_neg_labels[j][1]
        if metric=='rejection':
            fpr = data[0][j]
            tpr = data[1][j]
            threshold = data[2][j]
        
            roc_auc = auc(fpr, tpr)

            inv_fpr = []
            for i in fpr:
                inv_fpr.append(1/i) if i != 0 else inv_fpr.append(1/(3*min(fpr[fpr>0])))
            tnr = 1. - fpr
        elif metric == 'fraction':
            fraction = data[0][j]
            tpr = data[1][j]
            fpr = data[2][j]
            threshold = data[3][j]
            roc_auc = auc(fpr, tpr)
            tnr = 1. - fpr
        else:
            print('Error: metric must be either \'rejection\' or \'fraction\'.')
            return
        

        if metric == 'rejection':
            if plot_labels is None:
                line = ax.plot(tpr, inv_fpr,
                    label=f"{j:0.3f}, AUC {roc_auc:0.3f}",
                    linestyle=linestyle[j]  if linestyle is not None else None, linewidth=2,markerfacecolor=model_colors[j])
            else:
                line = ax.plot(tpr, inv_fpr,
                    label=f"{plot_labels[j]}, AUC {roc_auc:0.3f}",
                    linestyle=linestyle[j]  if linestyle is not None else None, linewidth=2,markerfacecolor=model_colors[j])
        else:
            if plot_labels is None:
                line = ax.plot(tpr, fraction,
                    label=f"{j:0.3f}, AUC {roc_auc:0.3f}",
                    linestyle=linestyle[j]  if linestyle is not None else None, linewidth=2,markerfacecolor=model_colors[j])
            else:
                line = ax.plot(tpr, fraction,
                    label=f"{plot_labels[j]}, AUC {roc_auc:0.3f}",
                    linestyle=linestyle[j]  if linestyle is not None else None, linewidth=2,markerfacecolor=model_colors[j])

        # Show coords of individual points near x = 0.2, 0.5, 0.8
        todo = {0.2: True, 0.5: True, 0.8: True}

        if annotate: 
            pbar = ProgressBar(widgets=['Find Critical Points: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
            ' ', ETA()], maxval=len(tpr))
            pbar.start()
            for i,xy in enumerate(zip(tpr, inv_fpr if metric=='rejection' else fraction, tnr)):
                pbar.update(i)
                xy = (round(xy[0], 4), round(xy[1], 4), round(xy[2], 4))
                xy_plot = (round(xy[0], 4), round(xy[1], 4))
                for point in todo.keys():
                    if xy[0] >= point and todo[point]:
                        ax.annotate('(%s, %s, %s)' % xy, xy=xy_plot, textcoords='data', fontsize=18, bbox=dict(boxstyle="square", fc="w"))
                        todo[point] = False
            pbar.finish()
        ax.grid(True, which='both', color='grey')

        if xlabel is None: xlabel = f'{legend_label_dict[label_0]} Signal Efficiency'
        if ylabel is None: ylabel = f'{legend_label_dict[label_1]} Background Rejection' if metric == 'rejection' else f'{legend_label_dict[label_1]} Background Rejection Fraction'

        ax.set_xlabel(xlabel, fontsize=20) 
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.legend(loc=leg_loc if leg_loc is not None else "upper right", prop={"size":20})
        if metric == 'rejection':
            ax.set_yscale('log')

        plt.margins(0.1)
        
    if png_name is not None: plt.savefig(os.path.join(os.getcwd(),png_name), bbox_inches='tight')    
                    
    return fpr, tpr, threshold, roc_auc

def prep_roc_data(softmaxes, labels, metric, softmax_index_dict, label_0, label_1, energies=None,threshold=None):
    """
    prep_roc_data(softmaxes, labels, metric, softmax_index_dict, label_0, label_1, energies=None,threshold=None)

    Purpose : Prepare data for plotting the ROC curves. If threshold is not none, filters 
    out events with energy greater than threshold. Returns true positive rates, false positive 
    rates, and thresholds for plotting the ROC curve, or true positive rates, rejection fraction,
    and thresholds, switched on 'metric'.

    Args: softmaxes           ... array of resnet softmax output, the 0th dim= sample size
          labels              ... 1D array of true label value, the length = sample size
          metric              ... string, name of metrix to use ('rejection' or 'fraction')
                                  for background rejection or background rejection fraction.
          softmax_index_dict  ... Dictionary pointing to label integer from particle name
          label_0 and label_1 ... Labels indicating which particles to use - label_0 is the positive label
          energies            ... 1D array of true event energies, the length = sample 
                                  size
          threshold           ... optional maximum to impose on energies, events with higher energy will be discarded (legacy)
    author: Calum Macdonald
    May 2020
    """    
    if threshold is not None and energies is not None:
        low_energy_idxs = np.where(np.squeeze(energies) < threshold)[0]
        rsoftmaxes = softmaxes[low_energy_idxs]
        rlabels = labels[low_energy_idxs]
        renergies = energies[low_energy_idxs]
    else:
        rsoftmaxes = softmaxes
        rlabels = labels
        renergies = energies

    (pos_softmaxes, neg_softmaxes), (pos_labels, neg_labels) = separate_particles([rsoftmaxes, rlabels], rlabels, softmax_index_dict, [label_0, label_1])    
    total_softmax = np.concatenate((pos_softmaxes, neg_softmaxes), axis=0)
    total_labels = np.concatenate((pos_labels, neg_labels), axis=0)
    assert total_labels.shape[0]==total_softmax.shape[0]

    if metric == 'rejection':
        return roc_curve(total_labels, total_softmax[:,softmax_index_dict[label_0]], pos_label=softmax_index_dict[label_0])
    else:
        fps, tps, thresholds = binary_clf_curve(total_labels,total_softmax[:,softmax_index_dict[label_0]], 
                                                pos_label=softmax_index_dict[label_0])
        fns = tps[-1] - tps
        tns = fps[-1] - fps
        tprs = tps / (tps + fns)
        rejection_fraction = tns / (tns + fps)
        fprs = fps / (fps + tns)
        return rejection_fraction, tprs, fprs, thresholds


def disp_multiple_learn_hist(locations,losslim=None,show=True,titles=None,best_only=False,leg_font=10,title_font=10):
    '''
    disp_multiple_learn_hist(locations,losslim=None,show=True,titles=None,best_only=False,leg_font=10,title_font=10)

    Plots a grid of learning histories.
    Args:
        locations               ... list of paths to directories of training dumps
        losslim                 ... limit of loss axis
        show                    ... bool, whether to show the plot
        titles                  ... list of titles for each plot in the grid
        best_only               ... bool, whether to plot only the points where best model was saved
        leg_font                ... legend font size
        title_font              ... take a guess!
    author: Calum Macdonald
    June 2020
    '''
    ncols = len(locations) if len(locations) < 3 else 3
    nrows = math.ceil(len(locations)/3)
    if nrows==1 and ncols==1: fig = plt.figure(facecolor='w',figsize=(12,12))
    else: fig = plt.figure(facecolor='w',figsize=(12,nrows*4))
    gs = gridspec.GridSpec(nrows,ncols,figure=fig)
    axes = []
    for i,location in enumerate(locations):
        train_log=location+'/log_train.csv'
        val_log=location+'/log_val.csv'        
        train_log_csv = pd.read_csv(train_log)
        val_log_csv  = pd.read_csv(val_log)
        
        disp_names = ['' for _ in range(len(locations))]
        if best_only:
            best_idxs = [0]
            best_epoch=0
            best_loss = val_log_csv.loss[0]
            for idx,loss in enumerate(val_log_csv.loss):
                if loss < best_loss: 
                    best_loss=loss
                    best_idxs.append(idx)
                    best_epoch=val_log_csv.epoch[idx]
            val_log_csv = val_log_csv.loc[best_idxs]
            if titles is not None:
                disp_names[i] = titles[i] + ", \n Best Val Loss ={loss:.4f}@Ep.{epoch:.2f}".format(loss=best_loss,epoch=best_epoch)
        else:
            disp_names=titles
        ax1=fig.add_subplot(gs[i],facecolor='w') if i ==0 else fig.add_subplot(gs[i],facecolor='w',sharey=axes[0])
        ax1.set_xlim(0,train_log_csv.epoch.max())
        axes.append(ax1)
        line11 = ax1.plot(train_log_csv.epoch, train_log_csv.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
        line12 = ax1.plot(val_log_csv.epoch, val_log_csv.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')
        if losslim is not None:
            ax1.set_ylim(None,losslim)
        if titles is not None:
            ax1.set_title(disp_names[i],size=title_font)
        ax2 = ax1.twinx()
        line21 = ax2.plot(train_log_csv.epoch, train_log_csv.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
        line22 = ax2.plot(val_log_csv.epoch, val_log_csv.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

        ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
        ax1.tick_params('x',colors='black',labelsize=18)
        ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
        ax1.tick_params('y',colors='b',labelsize=18)

        ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
        ax2.tick_params('y',colors='r',labelsize=18)
        ax2.set_ylim(0.,1.05)

        lines  = line11 + line12 + line21 + line22
        labels = [l.get_label() for l in lines]
        leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1,prop={'size':leg_font})
        leg_frame = leg.get_frame()
        leg_frame.set_facecolor('white')
    gs.tight_layout(fig)
    return fig


# Function to plot a grid of confusion matrices
def plot_multiple_confusion_matrix(label_arrays, prediction_arrays, class_names,titles=None):
    """
    plot_multiple_confusion_matrix(label_arrays, prediction_arrays, class_names,titles=None)    

    Plot the confusion matrix for a series of test outputs.
    
    Args: label_arrays        ... array of 1D arrays of true label value, the length = sample size
          predictions         ... array of 1D arrays of predictions, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories      
          titles              ... list of string titles for the confusion matrices
    author: Calum Macdonald
    May 2020
    """

    if len(label_arrays) >= 3:
        fig = plt.figure(facecolor='w',figsize=(16,8))
        gs = gridspec.GridSpec(math.ceil(len(label_arrays)/3),3,figure=fig)
    else:
        fig = plt.figure(facecolor='w',figsize=(10*len(label_arrays),8))
        gs = gridspec.GridSpec(1,len(label_arrays),figure=fig)
    axes = []

    for i,labels in enumerate(label_arrays):
        predictions = prediction_arrays[i]
        ax=fig.add_subplot(gs[i],facecolor='w')
        num_labels = len(class_names)
        max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
        assert max_value < num_labels
        mat,_,_,im = ax.hist2d(predictions, labels,
                               bins=(num_labels,num_labels),
                               range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

        # Normalize the confusion matrix
        mat = mat.astype("float") / mat.sum(axis=0)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=20) 

        ax.set_xticks(np.arange(num_labels))
        ax.set_yticks(np.arange(num_labels))
        ax.set_xticklabels(class_names,fontsize=20)
        ax.set_yticklabels(class_names,fontsize=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_xlabel('Prediction',fontsize=20)
        ax.set_ylabel('True Label',fontsize=20)
        if titles is not None: 
            ax.set_title(titles[i])

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                        ha="center", va="center", fontsize=20,
                        color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    gs.tight_layout(fig)
    return fig

def deprecated_load_fq_output(mapping_indices_path, fq_failed_idxs_path, test_idxs_path, cut_path, cut_list):
    '''
    load_fq_output(mapping_indices_path, test_idxs, cut_path, cut_list)
    
    Purpose : Load FiTQun output matching the desired 
    
    Args: mapping_indices_path ... path to .npz list of mapping indices for FiTQun
          fq_failed_idxs_path  ... path to .npz containing indices of events (in test set ordering) for which FiTQun failed to produce output 
          test_idxs_path       ... path to .npz containing indices in pointnet set - idx array must be titled 'test_idxs' in the archive
          cut_path             ... path to pointnet cuts npz file
          cut_list             ... list of cuts to be applied. Must be an array in the .npz pointed to be cut_path
    author: Calum Macdonald   
    August 2020
    '''
    # Load the fiTQun results

    # File paths for fiTQun results
    fiTQun_e_path     = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_e-.npz"
    fiTQun_mu_path    = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_mu-.npz"
    fiTQun_gamma_path = "/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_fiTQun_gamma.npz"

    # Load fiTQun results
    f_e = np.load(fiTQun_e_path, allow_pickle=True)
    f_mu = np.load(fiTQun_mu_path, allow_pickle=True)
    f_gamma = np.load(fiTQun_gamma_path, allow_pickle=True)

    fq_filename_original = (f_gamma['filename'],f_e['filename'],f_mu['filename'])
    fq_eventids_original = ( f_gamma['eventid'],f_e['eventid'], f_mu['eventid'])
    fq_flag_original = (f_gamma['flag'] ,f_e['flag'],f_mu['flag'])
    fq_nll_original = (f_gamma['nLL'],f_e['nLL'],f_mu['nLL'])
    fq_mom_original = (f_gamma['momentum'],f_e['momentum'],f_mu['momentum'])

    n_events = int(reduce(lambda x,y : x+y, list(map(lambda x : x.shape[0],fq_filename_original))))

    fq_rootfiles = np.empty(n_events,dtype=object)
    fq_eventids = np.zeros(n_events)
    fq_flag = np.empty((n_events,2))
    fq_nll = np.empty((n_events,2))
    fq_mom = np.empty((n_events, 2))

    fq_mapping_indices = np.load(mapping_indices_path,allow_pickle=True)['arr_0']

    fq_failed_idxs = np.load(fq_failed_idxs_path, allow_pickle = True)['failed_indices_pointing_to_h5_test_set'].astype(int)
    filtered_indices = np.load("/fast_scratch/WatChMaL/data/IWCD_fulltank_300_pe_idxs.npz", allow_pickle=True)
    test_filtered_indices = filtered_indices['test_idxs']
    stest_filtered_indices = np.delete(test_filtered_indices, fq_failed_idxs,0)

    idx_dic = {}
    for i, idx in enumerate(stest_filtered_indices):
        idx_dic[idx] = i

    test_idxs = np.load(test_idxs_path, allow_pickle=True)['test_idxs']

    keep_idxs = []
    stest_idxs = []
    i=0
    for idx in test_idxs:
        try:
            keep_idxs.append(idx_dic[idx])
            stest_idxs.append(idx)
        except KeyError:
            i+=1

    f = h5py.File("/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_full_tank_pointnet.h5", "r")
    original_labels = np.array(f['labels'])
    labels = np.array(original_labels[test_filtered_indices])
    f.close()
    slabels = np.delete(labels, fq_failed_idxs, 0)

    for i,ptype in enumerate(slabels):
        fq_rootfiles[i] = str(fq_filename_original[ptype][fq_mapping_indices[i]])
        fq_eventids[i] = fq_eventids_original[ptype][fq_mapping_indices[i]]
        fq_flag[i] = fq_flag_original[ptype][fq_mapping_indices[i]]
        fq_nll[i] = fq_nll_original[ptype][fq_mapping_indices[i]]
        fq_mom[i] = fq_mom_original[ptype][fq_mapping_indices[i]]
    fq_scores = np.zeros((fq_nll.shape[0],3))
    fq_scores[:,0] = fq_nll[:,1] - fq_nll[:,0]
    fq_scores[:,1] = fq_nll[:,1] - fq_nll[:,0]
    fq_scores[:,2] = fq_nll[:,0] - fq_nll[:,1]
    fq_labels = slabels

    fq_rootfiles = fq_rootfiles[keep_idxs]
    fq_eventids = fq_eventids[keep_idxs]
    fq_flag = fq_flag[keep_idxs]
    fq_scores = fq_scores[keep_idxs]
    fq_mom = fq_mom[keep_idxs]
    fq_labels = fq_labels[keep_idxs]

    fq_rootfiles = apply_cuts(fq_rootfiles, stest_idxs, cut_path, cut_list)
    fq_eventids = apply_cuts(fq_eventids, stest_idxs, cut_path, cut_list)
    fq_flag = apply_cuts(fq_flag, stest_idxs, cut_path, cut_list)
    fq_scores = apply_cuts(fq_scores, stest_idxs, cut_path, cut_list)
    fq_mom = apply_cuts(fq_mom, stest_idxs, cut_path, cut_list)
    fq_labels = apply_cuts(fq_labels, stest_idxs, cut_path, cut_list)

    return fq_rootfiles, fq_eventids, fq_flag, fq_scores, fq_mom, fq_labels

def apply_cuts(array, idxs, cut_path, cut_list):

    '''
    apply_cuts(array, indices, cut_path, cut_list)

    Purpose: Applies cuts to a given array, based on the given cut file and indices.

    Args:   array               ... 1d array of length n, that we wish to cut 
            idxs                ... 1d array of length n, where the ith entry gives the index of the event in the
                                    pointnet test set ordering corresponding to the ith entry in array
            cut_path            ... path to pointnet cuts npz file
            cut_list            ... list of cuts to be applied. Must be an array in the .npz pointed to be cut_path
    '''
    cut_file = np.load(cut_path, allow_pickle=True) 

    cut_arrays = []
    for cut in cut_list:
        assert cut in cut_file.keys(), f"Error, {cut} has no associated cut file"
        cut_arrays.append(cut_file[cut][idxs])

    combined_cut_array=np.array(list(map(lambda x : 1 if 1 in x else 0,  list(zip(*cut_arrays)))))
    cut_idxs = np.where(combined_cut_array==1)[0]    

    return np.delete(array, cut_idxs, 0)

def load_test_output_pn(location, cut_path, test_idxs, cut_list):
    """
    load_test_output_pn(location, cut_path, test_idxs, cut_list)
    
    Purpose : Load output of a test run on the full h5 test set, 
              using PN cut file method. 
    
    Args: location          ... string, path of the directory containing the test 
                                  output eg. '/home/cmacdonald/CNN/dumps/20200525_152544/test_validation_iteration_dump.npz'
          cut_path          ... string, path of cut file
          test_idxs         ... string, path of indices file used for test run
          cut_list          ... list of names of cuts, must be keys to files in the cut file
    author: Calum Macdonald   
    May 2020
    """
    test_dump_np = np.load(location, allow_pickle=True)
    cut_file = np.load(cut_path, allow_pickle=True) 

    cut_arrays = []
    for cut in cut_list:
        assert cut in cut_file.keys(), f"Error, {cut} has no associated cut file"
        cut_arrays.append(cut_file[cut][test_idxs])

    combined_cut_array=np.array(list(map(lambda x : 1 if 1 in x else 0,  list(zip(*cut_arrays)))))
    cut_idxs = np.where(combined_cut_array==1)[0]

    info_dict={}
    arr_names=['predicted_labels', 'softmax', 'labels', 'energies', 'rootfiles', 'eventids', 'angles']
    for arr_name in arr_names:
        info_dict[arr_name] = np.concatenate(list([batch_array for batch_array in test_dump_np[arr_name]]))

    for key in info_dict.keys():
        info_dict[key] = np.delete(info_dict[key], cut_idxs, 0)

    return info_dict

def load_test_output(location,index_path,remove_flagged=True, dset='noveto'):
    """
    load_test_output(location,index_path)
    
    (deprecated)
    Purpose : Load output of a test run on the full h5 test set, 
              remove FiTQun flagged/failed events, and return a dict of results.
    
    Args: location            ... string, path of the directory containing the test 
                                  output eg. '/home/cmacdonald/CNN/dumps/20200525_152544/test_validation_iteration_dump.npz'
          index_path          ... string, path of directory containing indices of FiTQun failed and flagged files
          remove_flagged      ... whether or not to remove flagged events, default True
          dset                ... 'noveto' or 'veto'
    author: Calum Macdonald   
    May 2020
    """
    test_dump_np = np.load(location, allow_pickle=True)
    
    res_predictedlabels = np.concatenate(list([batch_array for batch_array in test_dump_np['predicted_labels']]))
    res_softmaxes  = np.concatenate(list([batch_array for batch_array in test_dump_np['softmax']]))
    res_labels   = np.concatenate(list([batch_array for batch_array in test_dump_np['labels']]))
    res_energies = np.concatenate(list([batch_array for batch_array in test_dump_np['energies']]))
    res_rootfiles = np.concatenate(list([batch_array for batch_array in test_dump_np['rootfiles']]))
    res_eventids = np.concatenate(list([batch_array for batch_array in test_dump_np['eventids']]))
    res_angles = np.concatenate(list([batch_array for batch_array in test_dump_np['angles']]))
    
    if dset=='noveto':
        failed_idxs = np.load(os.path.join(index_path, 'fq_failed_idxs.npz'),allow_pickle=True)['failed_indices_pointing_to_h5_test_set'].astype(int)
        flagged_idxs = np.load(os.path.join(index_path, 'fq_flagged_idxs.npz'),allow_pickle=True)['arr_0'].astype(int)
        sres_predictedlabels = np.delete(res_predictedlabels,failed_idxs)
        sres_softmaxes  = np.delete(res_softmaxes,failed_idxs,0)
        sres_labels  = np.delete(res_labels,failed_idxs)
        sres_energies = np.delete(res_energies,failed_idxs)
        sres_rootfiles = np.delete(res_rootfiles,failed_idxs)
        sres_eventids = np.delete(res_eventids,failed_idxs)
        sres_angles = np.delete(res_angles,failed_idxs,0)

        if remove_flagged:    
            filtered_res_predictedlabels = np.delete(sres_predictedlabels,flagged_idxs)
            filtered_res_softmaxes  = np.delete(sres_softmaxes,flagged_idxs,0)
            filtered_res_labels  = np.delete(sres_labels,flagged_idxs)
            filtered_res_energies = np.delete(sres_energies,flagged_idxs)
            filtered_res_rootfiles = np.delete(sres_rootfiles,flagged_idxs)
            filtered_res_eventids = np.delete(sres_eventids,flagged_idxs)
            filtered_res_angles = np.delete(sres_angles,flagged_idxs,0)
            
            return{'filtered_predictions':filtered_res_predictedlabels,
                    'filtered_softmaxes':filtered_res_softmaxes,
                    'filtered_labels':filtered_res_labels,
                    'filtered_energies':filtered_res_energies,
                    'filtered_rootfiles':filtered_res_rootfiles,
                    'filtered_eventids':filtered_res_eventids,
                    'filtered_angles':filtered_res_angles          
                }
        else:
            return{'s_predictions':sres_predictedlabels,
                    's_softmaxes':sres_softmaxes,
                    's_labels':sres_labels,
                    's_energies':sres_energies,
                    's_rootfiles':sres_rootfiles,
                    's_eventids':sres_eventids,
                    's_angles':sres_angles          
                }
    elif dset=='vetoed':
            fq_cut_idxs = np.load(os.path.join(index_path, 'fq_cut_idxs_for_vetoed_set.npz'),allow_pickle=True)['fq_cut_idxs_for_vetoed_set'].astype(int)

            filtered_res_predictedlabels = np.delete(res_predictedlabels,fq_cut_idxs)
            filtered_res_softmaxes  = np.delete(res_softmaxes,fq_cut_idxs,0)
            filtered_res_labels  = np.delete(res_labels,fq_cut_idxs)
            filtered_res_energies = np.delete(res_energies,fq_cut_idxs)
            filtered_res_rootfiles = np.delete(res_rootfiles,fq_cut_idxs)
            filtered_res_eventids = np.delete(res_eventids,fq_cut_idxs)
            filtered_res_angles = np.delete(res_angles,fq_cut_idxs,0)
            
            return{'filtered_predictions':filtered_res_predictedlabels,
                    'filtered_softmaxes':filtered_res_softmaxes,
                    'filtered_labels':filtered_res_labels,
                    'filtered_energies':filtered_res_energies,
                    'filtered_rootfiles':filtered_res_rootfiles,
                    'filtered_eventids':filtered_res_eventids,
                    'filtered_angles':filtered_res_angles          
                }
    else:
        return{'filtered_predictions':res_predictedlabels,
                    'filtered_softmaxes':res_softmaxes,
                    'filtered_labels':res_labels,
                    'filtered_energies':res_energies,
                    'filtered_rootfiles':res_rootfiles,
                    'filtered_eventids':res_eventids,
                    'filtered_angles':res_angles          
                }

def parametrized_ray_point(x,y,z,theta,phi,t):
    '''
    parametrized_ray_point(x,y,z,theta,phi,t)

    Purpose: Find the point of a line departing (x,y,z) in direction specified by (theta,phi) and parametrized by t at
    given value of t. 
    Args: x, y, z           ... origin co-ordinates of line
          theta, phi        ... polar and azimuthal angles of departure
          t                 ... parameter giving desired point
    author: Calum Macdonald
    May 2020
    '''
    return x + np.sin(theta)*np.cos(phi)*t,y + np.sin(theta)*np.sin(phi)*t, z + np.cos(theta)*t


def distance_to_wall(position, angle):
    """
    distance_to_wall(position, angle)
    
    Purpose : Calculate distance from event origin to IWCD wall along particle trajectory.
    
    Args: position            ... array of [x, y, z] co-ordinates of event origin
          angle               ... array of [theta, phi] angles of departure
    author: Calum Macdonald
    May 2020
    """
    x = position[0]
    y = position[2]
    z = position[1]
    theta = angle[0]
    phi = angle[1]
    no_radial=False
    sols = []
    #Solve for intersections of parametrized path with the cylinder and caps, keep only positive parameter solns
    try:
        shared_expression = np.sqrt(-np.sin(theta)**2*(-275282+(x**2 + y**2)
                                 + (y**2 - x**2)*np.cos(2*phi)-2*x*y*np.sin(2*phi)))/(np.sin(theta)*np.sqrt(2))
    except:
        no_radial=True
    if not no_radial:
        try:
            radial_parameter_sol_1 = -1/np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi)
                                     +shared_expression)
            if radial_parameter_sol_1 > 0: sols.append(radial_parameter_sol_1)
        except:
            pass
        try:
            radial_parameter_sol_2 = 1/np.sin(theta)*(-x*np.cos(phi)-y*np.sin(phi)
                                     +shared_expression)
            if radial_parameter_sol_2 > 0: sols.append(radial_parameter_sol_2)
        except:
            pass
    try:
        cap_parameter_sol_top = (521 - z)/np.cos(theta)
        cap_parameter_sol_bottom = -(521+z)/np.cos(theta)
        if cap_parameter_sol_top > 0: sols.append(cap_parameter_sol_top)
        if cap_parameter_sol_bottom > 0: sols.append(cap_parameter_sol_bottom)
    except:
        pass
    sols = np.sort(sols)
    x_int,y_int,z_int = parametrized_ray_point(x,y,z,theta,phi,sols[0])
    return np.sqrt((x-x_int)**2+(y-y_int)**2+(z-z_int)**2)


def plot_compare_dists(dists,dist_idxs_to_compare,dist_idxs_reference,
                       labels,axes=None,colors=None,bins=20,
                       title=None, ratio_range=None,xlabel=None,
                       linestyle=None):
    '''
    Plot distributions and plot their ratio.
    Args:
        dists                   ... list of 1d arrays
        dist_idxs_to_compare    ... list of indices of distributions to use as numerator 
                                    in the ratio plot
        dist_idxs_reference     ... list of indices of distributions to use as denominator
                                    in the ratio plot
        labels                  ... list of labels for each distribution
        axes                    ... optional, list of two matplotlib.pyplot.axes on which 
                                    to place the plots
        colors                  ... list of colors to use for each distribution
        bins                    ... number of bins to use in histogram
        title                   ... plot title
        ratio_range             ... range of distribution range to plot
        xlabel                  ... x-axis label
        linestyle               ... list of linestyles to use for each distribution
    author: Calum Macdonald
    June 2020
    '''
    ret = False
    if axes is None:
        fig, axes = plt.subplots(2,1,figsize=(12,12))
        ret = True
    axes = axes.flatten()
    ax = axes[0]
    ns, bins, patches = ax.hist(dists, weights=[np.ones(len(dists[i]))*1/len(dists[i]) for i in range(len(dists))], 
                                label=labels,histtype=u'step',bins=bins,color=colors,alpha=0.8)
    if linestyle is not None:
        for i,patch_list in enumerate(patches):
            for patch in patch_list:
                patch.set_linestyle(linestyle[i])
            
    ax.legend()
    if title is not None: ax.set_title(title)
    ax2 = axes[1]
    for i,idx in enumerate(dist_idxs_to_compare):
        lines = ax2.plot(bins[:-1],     
                 ns[idx] / ns[dist_idxs_reference[i]], 
                 alpha=0.8,label='{} to {}'.format(labels[idx],labels[dist_idxs_reference[i]]))
        lines[0].set_color(patches[idx][0].get_edgecolor())
        lines[0].set_drawstyle('steps')
    if ratio_range is not None: ax2.set_ylim(ratio_range)
    ax2.legend()
    ax2.set_title('Ratio of Distributions')
    lines = ax2.plot(bins[:-1],np.ones(len(bins)-1),color='k',alpha=0.5)
    lines[0].set_linestyle('-.')
    if xlabel is not None: 
        ax.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
    if ret: return fig


def plot_2d_ratio(dist_1_x,dist_1_y,dist_2_x, dist_2_y,bins=(150,150),fig=None,ax=None,
                  title=None, xlabel=None, ylabel=None, ratio_range=None):
    '''
    Plots the 2d ratio between the 2d histograms of two distributions.
    Args:
        dist_1_x:               ... 1d array of x-values of distribution 1 of length n
        dist_1_y:               ... 1d array of y-values of distribution 1 of length n
        dist_2_x:               ... 1d array of x-values of distribution 2 of length n
        dist_2_y:               ... 1d array of y-values of distribution 2 of length n
        bins:                   ... tuple of integer numbers of bins in x and y 
        fig                     ... figure on which to plot
        ax                      ... axis on which to plot ratio
        title                   ... str, title
        xlabel                  ... str, x-axis label
        ylabel                  ... str, y-axis label
        ratio_range             ... range of ratios to display
    author: Calum Macdonald
    May 2020
    '''
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(8,8))
    bin_range = [[np.min([np.min(dist_1_x),np.min(dist_2_x)]),np.max([np.max(dist_1_x),np.max(dist_2_x)])],
             [np.min([np.min(dist_1_y),np.min(dist_2_y)]),np.max([np.max(dist_1_y),np.max(dist_2_y)])]]
    ns_1, xedges, yedges = np.histogram2d(dist_1_x,dist_1_y,bins=bins,density=True,range=bin_range)
    ns_2,_,_ = np.histogram2d(dist_2_x,dist_2_y,bins=bins,density=True,range=bin_range)
    ratio = ns_1/ns_2
    ratio = np.where((ns_2==0) & (ns_1==0),1,ratio)
    ratio = np.where((ns_2==0) & (ns_1!=0),10,ratio)
    pc = ax.pcolormesh(xedges, yedges, np.swapaxes(ratio,0,1),vmin=ratio_range[0],vmax=ratio_range[1],cmap="RdBu_r")
    fig.colorbar(pc, ax=ax)
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig


a = np.ones((100,100))

[i for i in itertools.product(range(3),range(3))]

def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    '''
        SOURCE: Scikit.metrics internal usage tool
    Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    '''

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
                             classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]



def plot_binned_performance(softmaxes, labels, binning_features, binning_label,efficiency, bins, index_dict, 
                            label_0, label_1, metric='purity',fixed='rejection',ax=None,marker='',color=None,legend_label_dict=None,title_note=''):
    '''
    Plots the purity as a function of a physical parameter in the dataset, at a fixed signal efficiency (true positive rate).
    Args:
        softmaxes                      ... 2d array with first dimension n_samples
        labels                         ... 1d array of true event labels
        binning_features               ... 1d array of features for generating bins, eg. energy
        binning_label                  ... name of binning feature, to be used in title and xlabel
        efficiency                     ... signal efficiency per bin, to be fixed
        bins                           ... either an integer (number of evenly spaced bins) or list of n_bins+1 edges
        index_dict                     ... dictionary of particle string keys and values corresponding to labels in 'labels'
        label_0                        ... string, positive particle label, must be key of index_dict
        label_1                        ... string, negative particle label, must be key of index_dict
        metric                         ... string, metric to plot ('purity' for signal purity, 'rejection' for rejection fraction, 'efficiency' for signal efficiency)
        ax                             ... axis on which to plot
        color                          ... marker color
        marker                         ... marker type
        legend_label_dict              ... dictionary of display symbols for each string label, to use for displaying pretty characters
        title_note                     ... string to append to the plot title
    author: Calum Macdonald
    June 2020
    '''
    if legend_label_dict is None:
            legend_label_dict={}
            for name in [label_0, label_1]:
                legend_label_dict[name] = name
    label_size = 14

    assert binning_features.shape[0] == softmaxes.shape[0], 'Error: binning_features must have same length as softmaxes'

    #bin by whatever feature
    if isinstance(bins, int):
        _,bins = np.histogram(binning_features, bins=bins)
    bins = bins[0:-1]
    bin_assignments = np.digitize(binning_features, bins)
    bin_data = []
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        this_bin_idxs = np.where(bin_assignments==bin_num)[0]
        bin_data.append({'softmaxes':softmaxes[this_bin_idxs], 'labels' : labels[this_bin_idxs], 'n' : this_bin_idxs.shape[0]})

    #compute efficiency, thresholds, purity per bin
    bin_metrics = []
    for bin_idx, data in enumerate(bin_data):
        (softmaxes_0,softmaxes_1),(labels_0,labels_1) = separate_particles([data['softmaxes'],data['labels']],data['labels'],index_dict,desired_labels=[label_0,label_1])
        fps, tps, thresholds = binary_clf_curve(np.concatenate((labels_0,labels_1)),np.concatenate((softmaxes_0,softmaxes_1))[:,index_dict[label_0]], 
                                                pos_label=index_dict[label_0])
        fns = tps[-1] - tps
        tns = fps[-1] - fps
        efficiencies = tps/(tps + fns)
        operating_point_idx = (np.abs(efficiencies - efficiency)).argmin()
        if metric == 'purity': performance = tps/(tps + fps)
        elif metric == 'rejection': performance = tns / (tns + fps)
        elif metric == 'inverse fpr': performance = np.where(fps != 0, (fps +tns) / fps, fps+tns)
        bin_metrics.append((efficiencies[operating_point_idx], performance[operating_point_idx], np.sqrt(tns[operating_point_idx])/(tns[operating_point_idx] + fps[operating_point_idx])))
    bin_metrics = np.array(bin_metrics)

    bin_centers = [(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins)-1)]
    bin_centers.append((np.max(binning_features) - bins[-1])/2 + bins[-1])

    if metric == 'purity':
        metric_name = '{}-{} Signal Purity'.format(label_0,label_1) 
    elif metric=='rejection': metric_name =  '{} Rejection Fraction'.format(legend_label_dict[label_1]) 
    else: metric_name = '{} Rejection'.format(legend_label_dict[label_1]) 
    title = '{} \n vs {} At Bin {} Signal Efficiency {}{}'.format(metric_name, binning_label, legend_label_dict[label_0], efficiency,title_note)
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        plt.errorbar(bin_centers,bin_metrics[:,1],yerr=bin_metrics[:,2],fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        plt.ylabel(metric_name, fontsize=label_size)
        plt.xlabel(binning_label, fontsize=label_size)
        plt.title(title)

    else:
        ax.errorbar(bin_centers,bin_metrics[:,1],yerr=bin_metrics[:,2],fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        ax.set_ylabel(metric_name, fontsize=label_size)
        ax.set_xlabel(binning_label, fontsize=label_size)
        ax.set_title(title)
        if metric=='inverse fpr': ax.set_yscale('log')

def plot_fitqun_binned_performance(scores, labels, true_momentum, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, true_mom_bins=20, 
                            ax=None,marker='o',color='k',title_note='',metric='efficiency',yrange=None):
    '''
    plot_fitqun_binned_performance(scores, labels, true_momentum, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, true_mom_bins=20, 
                            ax=None,marker='o',color='k',title_note='',metric='efficiency',yrange=None)
        Purpose: Re-create official FiTQun plots.

        Args:
            scores                  ... network scores for each class
            labels                  ... 1d array of labels
            true_momentum           ... 1d array of event true momentum
            reconstructed_momentum  ... 1d array of FQ reconstructed momentum
            fpr_fixed_point         ... fixed false-positive rate for FQ recons. mom. bins
            index_dict              ... dictionary with 'e', 'mu' keys pointing to corresponding integer labels
            recons_mom_bin_size     ... size of reconstructed mom. bin
            true_mom_bins           ... number of true momentum bins
            ax                      ... axis to plot on
            marker                  ... marker for plor
            color                   ... curve color
            title_note              ... string to append to title
            metric                  ... 'efficiency' will give signal efficiency, any other will give FPR
            yrange                  ... range for the y axis
    '''
    label_size = 14

    #remove gamma events
    scores, labels, true_momentum, reconstructed_momentum = separate_particles([scores, labels, true_momentum, reconstructed_momentum],labels,index_dict,desired_labels=['e','mu'])
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    true_momentum = np.concatenate(true_momentum)
    reconstructed_momentum = np.concatenate(reconstructed_momentum)

    #bin by reconstructed momentum
    bins = [0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))]   
    bins = bins[0:-1]
    recons_mom_bin_assignments = np.digitize(reconstructed_momentum, bins)
    recons_mom_bin_idxs_list = [[]]*len(bins)
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        recons_mom_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments==bin_num)[0]

    #compute threshold giving fixed fpr per reconstructed energy bin
    thresholds_per_event = np.ones_like(labels, dtype=float)
    for bin_idx, bin_idxs in enumerate(recons_mom_bin_idxs_list):        
        if bin_idxs.shape[0] > 0:
            fps, tps, thresholds = binary_clf_curve(labels[bin_idxs],scores[bin_idxs], 
                                                    pos_label=index_dict['e'])
            fns = tps[-1] - tps
            tns = fps[-1] - fps
            fprs = fps/(fps + tns)
            operating_point_idx = (np.abs(fprs - fpr_fixed_point)).argmin()
            thresholds_per_event[bin_idxs] = thresholds[operating_point_idx]

    #bin by true momentum
    ns,bins = np.histogram(true_momentum, bins=true_mom_bins, range=(200., np.max(true_momentum)) if metric=='mu fpr' else (0,1000))
    bins = bins[0:-1]
    true_mom_bin_assignments = np.digitize(true_momentum, bins)
    true_mom_bin_idxs_list = [[]]*len(bins)
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        true_mom_bin_idxs_list[bin_idx]=np.where(true_mom_bin_assignments==bin_num)[0]

    #find metrics for each true momentum bin
    bin_metrics=[]
    for bin_idxs in true_mom_bin_idxs_list:
        pred_pos_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] > 0)[0]
        pred_neg_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] < 0)[0]
        fp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['mu'] )[0].shape[0]
        tp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['e'] )[0].shape[0]
        fn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['e'] )[0].shape[0]
        tn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['mu'] )[0].shape[0]
        if metric=='efficiency':
            bin_metrics.append(tp/(tp+fn))
        else:
            bin_metrics.append(fp/(fp + tn))

    #plot metrics
    bin_centers = [(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins)-1)]
    bin_centers.append((np.max(true_momentum) - bins[-1])/2 + bins[-1] if metric=='mu fpr' else (1000 - bins[-1])/2 + bins[-1])
    metric_name = 'e- Signal Efficiency' if metric== 'efficiency' else '\u03BC- Mis-ID Rate'
    title = '{} \n vs True Momentum At Reconstructed Momentum Bin \u03BC- Mis-ID Rate of {}%{}'.format(metric_name, fpr_fixed_point*100, title_note)
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        plt.errorbar(bin_centers,bin_metrics,yerr=np.zeros_like(bin_metrics),fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        plt.ylabel(metric_name)
        plt.xlabel("True Momentum (MeV/c)", fontsize=label_size)
        if yrange is not None: plt.ylim(yrange)
        plt.title(title)
    else:
        ax.errorbar(bin_centers[:50],bin_metrics[:50],yerr=np.zeros_like(bin_metrics[:50]),fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        nax = ax.twinx()
        # nax.bar(bin_centers,ns,fill=False,width=bins[3]-bins[2])
        ax.set_ylabel(metric_name)
        ax.set_xlabel("True Momentum (MeV/c)", fontsize=label_size)
        if yrange is not None: ax.set_ylim(yrange) 
        ax.set_title(title)

    return true_momentum, thresholds_per_event


def plot_response(softmaxes, labels, particle_names, index_dict,linestyle=None,bins=None,fig=None,axes=None,legend_locs=None,fitqun=False,
                  extra_panes=[], xlim=None,label_size=14, legend_label_dict=None):
    '''
    Plots classifier softmax outputs for each particle type.
    Args:
        softmaxes                    ... 2d array with first dimension n_samples
        labels                       ... 1d array of particle labels to use in every output plot, or list of 4 lists of particle names to use in each respectively
        particle_names               ... list of string names of particle types to plot. All must be keys in 'index_dict' 
        index_dict                   ... dictionary of particle labels, with string particle name keys and values corresponsing to 
                                         values taken by 'labels'
        bins                         ... optional, number of bins for histogram
        fig, axes                    ... optional, figure and axes on which to do plotting (use to build into bigger grid)
        legend_locs                  ... list of 4 strings for positioning the legends
        fitqun                       ... designate if the given scores are from fitqun
        xlim                        ... limit the x-axis
        label_size                  ... font size
        legend_label_dict           ... dictionary of display symbols for each string label, to use for displaying pretty characters
    author: Calum Macdonald
    June 2020
    '''
    if legend_label_dict is None:
        legend_label_dict={}
        for name in particle_names:
            legend_label_dict[name] = name
    legend_size=label_size

    num_panes = softmaxes.shape[1]+len(extra_panes)

    if axes is None:
        fig,axes = plt.subplots(1,num_panes,figsize=(5*num_panes,5)) if not fitqun else plt.subplots(1,1,figsize=(7,7))
    label_dict = {value:key for key, value in index_dict.items()}

    softmaxes_list = separate_particles([softmaxes], labels, index_dict, [name for name in index_dict.keys()])[0]

    if isinstance(particle_names[0],str):
        particle_names = [particle_names for _ in range(num_panes)]

    if fitqun:
        ax = axes
        density = False
        for i in [index_dict[particle_name] for particle_name in particle_names[1]]:
            _,bins,_ = ax.hist(softmaxes_list[i][:,1],
                        label=legend_label_dict[label_dict[i]],range=xlim,
                        alpha=0.7,histtype=u'step',bins=bins,density=density,
                        linestyle=linestyle[i],linewidth=2)    
            ax.legend(loc=legend_locs[0] if legend_locs is not None else 'best', fontsize=legend_size)
            ax.set_xlabel('e-muon nLL Difference')
            ax.set_ylabel('Normalized Density' if density else 'N Events', fontsize=label_size)
    else:
        for output_idx,ax in enumerate(axes[:softmaxes.shape[1]]):
            for i in [index_dict[particle_name] for particle_name in particle_names[output_idx]]:
                ax.hist(softmaxes_list[i][:,output_idx],
                        label=f"{legend_label_dict[label_dict[i]]} Events",
                        alpha=0.7,histtype=u'step',bins=bins,density=True,
                        linestyle=linestyle[i],linewidth=2)            
            ax.legend(loc=legend_locs[output_idx] if legend_locs is not None else 'best', fontsize=legend_size)
            ax.set_xlabel('P({})'.format(legend_label_dict[label_dict[output_idx]]), fontsize=label_size)
            ax.set_ylabel('Normalized Density', fontsize=label_size)
            ax.set_yscale('log')
        ax = axes[-1]
        for n, extra_pane_particle_names in enumerate(extra_panes):
            pane_idx = softmaxes.shape[1]+n
            ax=axes[pane_idx]
            for i in [index_dict[particle_name] for particle_name in particle_names[pane_idx]]:
                    ax.hist(reduce(lambda x,y : x+y, [softmaxes_list[i][:,index_dict[pname]] for pname in extra_pane_particle_names]),
                            label=legend_label_dict[particle_names[-1][i]],
                            alpha=0.7,histtype=u'step',bins=bins,density=True,
                            linestyle=linestyle[i],linewidth=2)         
            ax.legend(loc=legend_locs[-1] if legend_locs is not None else 'best', fontsize=legend_size)
            ax.set_xlabel('P({}) + P({})'.format(legend_label_dict['gamma'],legend_label_dict['e']), fontsize=label_size)
            ax.set_ylabel('Normalized Density', fontsize=label_size)
            ax.set_yscale('log')
    plt.tight_layout()
    return fig

def rms(arr):
    '''
    Returns RMS value of the array.
    Args:
        arr                         ... 1d array of numbers
    author: Calum Macdonald
    June 2020
    '''
    return math.sqrt(reduce(lambda a, x: a + x * x, arr, 0) / len(arr))

def plot_binned_response(softmaxes, labels, particle_names, binning_features, binning_label,efficiency, bins, p_bins, index_dict, extra_panes=None, log_scales=[], legend_label_dict=None, wrap_size=35):
    '''
    Plot softmax response, binned in a feature of the event.
    Args:
        softmaxes                   ... 2d array of softmax output, shape (nsamples, noutputs)
        labels                      ... 1d array of labels, length n_samples
        particle_names              ... string particle names for which to plot the response, must be keys of index_dict
        binning_features            ... 1d array of feature to use in binning, length n_samples
        binning_label               ... string, name of binning feature to use in title and x-axis label
        efficiency                  ... bin signal efficiency to fix
        bins                        ... number of bins to use in feature histogram
        p_bins                      ... number of bins to use in probability density histogram
        index_dict                  ... dictionary of particle labels, must have all of particle_names as keys, pointing to values taken by 'labels'
        extra_panes                 ... list of lists of particle names to combine into an "extra output" e.g. [["e", "gamma"]] adds the P(e-)+P(gamma) pane
        log_scales                  ... indices of axes.flatten() to which to apply log color scaling
        legend_label_dict           ... dictionary of display symbols for each string label, to use for displaying pretty characters
        wrap_size                   ... width of box to wrap title into
    author: Calum Macdonald
    June 2020
    '''

    if legend_label_dict is None:
        legend_label_dict = {}
        for name in particle_names: legend_label_dict[name] = name

    noutputs = softmaxes.shape[1]
    if extra_panes is None: extra_panes=[]

    label_size = 18
    fig, axes = plt.subplots(len(particle_names),noutputs+len(extra_panes),figsize=(12*len(particle_names),12*noutputs+len(extra_panes)))

    log_axes = axes.flatten()[log_scales]

    #bin by whatever feature
    if isinstance(bins, int):
        _,bins = np.histogram(binning_features, bins=bins)
    b_bin_centers = [bins[i] + (bins[i+1]-bins[i])/2 for i in range(bins.shape[0]-1)]
    binning_edges=bins
    bins = bins[0:-1]
    bin_assignments = np.digitize(binning_features, bins)
    bin_data = []
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        this_bin_idxs = np.where(bin_assignments==bin_num)[0]
        bin_data.append({'softmaxes':softmaxes[this_bin_idxs], 'labels' : labels[this_bin_idxs]})
    
    edges = None
    for output_idx in range(noutputs):
        for particle_idx, particle_name in [(index_dict[particle_name], particle_name) for particle_name in particle_names]:
            ax = axes[particle_idx][output_idx]
            data = np.ones((len(bins), len(p_bins) if not isinstance(p_bins, int) else p_bins))
            means = []
            stddevs = []
            for bin_idx, bin in enumerate(bin_data):
                relevant_softmaxes = separate_particles([bin['softmaxes']],bin['labels'], index_dict, desired_labels=[particle_name])[0][0]

                if edges is None: ns, edges = np.histogram(relevant_softmaxes[:,output_idx], bins=p_bins,density=True,range=(0.,1.))
                else: ns, _ = np.histogram(relevant_softmaxes[:,output_idx], bins=edges,density=True)

                data[bin_idx, :] = ns
                p_bin_centers = [edges[i] + (edges[i+1]-edges[i])/2 for i in range(edges.shape[0]-1)]
                means.append(np.mean(relevant_softmaxes[:,output_idx]))
                stddevs.append(np.std(relevant_softmaxes[:,output_idx]))                

            if ax in log_axes:
                min_value = np.unique(data)[1]
                data = np.where(data==0, min_value, data)
            mesh = ax.pcolormesh(binning_edges, edges, np.swapaxes(data,0,1),norm=colors.LogNorm() if ax in log_axes else None)
            fig.colorbar(mesh,ax=ax)
            ax.errorbar(b_bin_centers, means, yerr = stddevs, fmt='k+', ecolor='k', elinewidth=0.5, capsize=4, capthick=1.5)
            ax.set_xlabel(binning_label,fontsize=label_size)
            ax.set_ylabel('P({})'.format(legend_label_dict[particle_names[output_idx]]),fontsize=label_size)
            ax.set_ylim([0,1])
            ax.set_title('\n'.join(wrap('P({}) Density For {} Events vs {}'.format(legend_label_dict[particle_names[output_idx]],legend_label_dict[particle_name],binning_label),wrap_size)),fontsize=label_size)

    for n, extra_pane_particle_names in enumerate(extra_panes):
        for particle_idx, particle_name in [(index_dict[particle_name], particle_name) for particle_name in particle_names]:
                ax = axes[particle_idx][n+noutputs]
                data = np.ones((len(bins), len(p_bins) if not isinstance(p_bins, int) else p_bins))
                means = []
                stddevs = []
                for bin_idx, bin in enumerate(bin_data):
                    relevant_softmaxes = separate_particles([bin['softmaxes']],bin['labels'], index_dict, desired_labels=[particle_name])[0][0]
                    extra_output = reduce(lambda x,y : x+y, [relevant_softmaxes[:,index_dict[pname]] for pname in extra_pane_particle_names])
                    ns, _ = np.histogram(extra_output, bins=edges,density=True)
                    data[bin_idx, :] = ns
                    p_bin_centers = [edges[i] + (edges[i+1]-edges[i])/2 for i in range(edges.shape[0]-1)]
                    means.append(np.mean(extra_output))
                    stddevs.append(np.std(extra_output))
                if ax in log_axes:
                    min_value = np.unique(data)[1]
                    data = np.where(data==0, min_value, data)
                mesh = ax.pcolormesh(binning_edges,edges, np.swapaxes(data,0,1),norm=colors.LogNorm() if ax in log_axes else None)
                fig.colorbar(mesh,ax=ax)
                ax.set_ylim([0,1])
                ax.errorbar(b_bin_centers, means, yerr = stddevs, fmt='k+', ecolor='k', elinewidth=0.5, capsize=4, capthick=1.5)
                ax.set_xlabel(binning_label,fontsize=label_size)
                extra_output_label = reduce(lambda x,y: x + ' + ' + f"P({y})", [f'P({legend_label_dict[name]})' if i==0 else legend_label_dict[name] for i, name in enumerate(extra_pane_particle_names)])
                ax.set_ylabel(extra_output_label,fontsize=label_size)
                ax.set_title('\n'.join(wrap('{} Density For {} Events vs {}'.format(extra_output_label, legend_label_dict[particle_name],binning_label),wrap_size)),fontsize=label_size)


def separate_particles(input_array_list,labels,index_dict,desired_labels=['gamma','e','mu']):
    '''
    Separates all arrays in a list by indices where 'labels' takes a certain value, corresponding to a particle type.
    Assumes that the arrays have the same event order as labels. Returns list of tuples, each tuple contains section of each
    array corresponsing to a desired label.
    Args:
        input_array_list            ... list of arrays to be separated, must have same length and same length as 'labels'
        labels                      ... list of labels, taking any of the three values in index_dict.values()
        index_dict                  ... dictionary of particle labels, must have 'gamma','mu','e' keys pointing to values taken by 'labels', 
                                        unless desired_labels is passed
        desired_labels              ... optional list specifying which labels are desired and in what order. Default is ['gamma','e','mu']
    author: Calum Macdonald
    June 2020
    '''
    idxs_list = [np.where(labels==index_dict[label])[0] for label in desired_labels]

    separated_arrays = []
    for array in input_array_list:
        separated_arrays.append(tuple([array[idxs] for idxs in idxs_list]))

    return separated_arrays

def collapse_test_output(softmaxes, labels, index_dict,predictions=None,ignore_type=None):
    '''
    Collapse gamma class into electron class to allow more equal comparison to FiTQun.
    Args:
        softmaxes                  ... 2d array of dimension (n,3) corresponding to softmax output
        labels                     ... 1d array of event labels, of length n, taking values in the set of values of 'index_dict'
        index_dict                 ... Dictionary with keys 'gamma','e','mu' pointing to the corresponding integer
                                       label taken by 'labels'
        predictions                ... 1d array of event type predictions, of length n, taking values in the 
                                       set of values of 'index_dict'   
        ignore_type                ... single string, name of event type to exclude                     
    '''
    if ignore_type is not None:
        keep_indices = np.where(labels!=index_dict[ignore_type])[0]
        softmaxes = softmaxes[keep_indices]
        labels = labels[keep_indices]
        if predictions is not None: predictions = predictions[keep_indices]

    new_labels = np.ones((softmaxes.shape[0]))*index_dict['e']
    new_softmaxes = np.zeros((labels.shape[0], 3))
    if predictions is not None:
        new_predictions = np.ones_like(predictions) * index_dict['e']
    for idx, label in enumerate(labels):
            if index_dict["mu"] == label: new_labels[idx] = index_dict["mu"]
            new_softmaxes[idx,:] = [0,softmaxes[idx][0] + softmaxes[idx][1], softmaxes[idx][2]]
            if predictions is not None:
                if predictions[idx]==index_dict['mu']: new_predictions[idx] = index_dict['mu']

    if predictions is not None: return new_softmaxes, new_labels, new_predictions
    return new_softmaxes, new_labels
