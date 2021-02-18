import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.utils.extmath import stable_cumsum
from progressbar import *

def load_test_output_pn(location, cut_path, test_idxs, cut_list):
    test_dump_np = np.load(location, allow_pickle=True)
    cut_file = np.load(cut_path, allow_pickle=True) 

    cut_arrays = []
    for cut in cut_list:
        assert cut in cut_file.keys(), f"Error, {cut} has no associated cut file"
        cut_arrays.append(cut_file[cut][test_idxs])
        print(1 in cut_file[cut][test_idxs])

    combined_cut_array=np.array(list(map(lambda x : 1 if 1 in x else 0,  list(zip(*cut_arrays)))))
    cut_idxs = np.where(combined_cut_array==1)[0]

    info_dict={}
    arr_names=['predicted_labels', 'softmax', 'labels', 'energies', 'rootfiles', 'eventids', 'angles']
    for arr_name in arr_names:
        info_dict[arr_name] = np.concatenate(list([batch_array for batch_array in test_dump_np[arr_name]]))

    print("data shape: ", np.array(info_dict['predicted_labels']).shape)
    print("test indices shape: ", test_idxs.shape)

    for key in info_dict.keys():
        info_dict[key] = np.delete(info_dict[key], cut_idxs, 0)

    return info_dict

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

def test_collapse_test_output(softmaxes, labels, index_dict,predictions=None,ignore_type=None):
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

    new_labels = np.ones((softmaxes.shape[0]))*index_dict['$e$']
    new_softmaxes = np.zeros((labels.shape[0], 3))

    if predictions is not None:
        new_predictions = np.ones_like(predictions) * index_dict['$e$']
    
    for idx, label in enumerate(labels):
            if index_dict["$\mu$"] == label: new_labels[idx] = index_dict["$\mu$"]
            new_softmaxes[idx,:] = [0,softmaxes[idx][0] + softmaxes[idx][1], softmaxes[idx][2]]
            if predictions is not None:
                if predictions[idx]==index_dict['$\mu$']: new_predictions[idx] = index_dict['$\mu$']

    if predictions is not None: return new_softmaxes, new_labels, new_predictions
    
    return new_softmaxes, new_labels


def collapse_test_output(softmaxes, labels, index_dict, plot_list, vs_list, predictions=None,ignore_type=None):
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
    """
    if ignore_type is not None:
        keep_indices = np.where(labels!=index_dict[ignore_type])[0]
        softmaxes = softmaxes[keep_indices]
        labels = labels[keep_indices]
        if predictions is not None: predictions = predictions[keep_indices]
    """
    independent_particle_labels = np.array([index_dict[particle_name] for particle_name in plot_list])
    dependent_particle_labels = np.array([index_dict[particle_name] for particle_name in vs_list])

    # Initialize new set of output labels
    new_labels = np.ones((softmaxes.shape[0]))

    # Initialize output softmax array with collapsed dimension
    new_softmaxes = np.zeros((labels.shape[0], 3))

    """
    if predictions is not None:
        new_predictions = np.ones_like(predictions) * index_dict['$e$']
    """

    for idx, label in enumerate(labels):
        if label in dependent_particle_labels:
            new_labels[idx] = 1
        new_softmaxes[idx,:] = [0, np.sum((softmaxes[idx])[independent_particle_labels]), np.sum((softmaxes[idx])[dependent_particle_labels])]
        """
        if predictions is not None:
            if predictions[idx]==index_dict['$\mu$']: new_predictions[idx] = index_dict['$\mu$']
        """
    """
    if predictions is not None: return new_softmaxes, new_labels, new_predictions
    """

    return new_softmaxes, new_labels
