#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def classifier_plot(labels, predictions, norm_ax=0, do_return=False):
    '''
    Plot the predictions of a classifier vs. the actual class labels
    
    labels: ndarray
        The actual classlabel
        
    predictions: ndarray of the same shape as "labels"
        The corresponding predictions given by the classifier
        
    norm_ax: int: 0 or 1, default=0
        If norm_ax=0, then the numbers in the plot give the probability P(real_class=y-value | predicted_class=x-value), i.e. each column is normalized and sums up to 1 (approx., due to rounding errors).
    
        If norm_ax=1, then the numbers in the plot give the probability P(predicted_class=x-value | real_class=y-value), i.e. each row is normalized and sums up to 1 (approx., due to rounding errors).
    
    do_return: bool, default=False
        Shall the fig and ax be returned?
    
    returns: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot if do_return=True, else None
        The figure and axes of the plot
    '''
    labels     = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    
    num_Classes = int(np.max(labels) + 1)
    num_Samples = len(labels)
    count = np.zeros((num_Classes,num_Classes))
    for i in range(num_Samples):
        realclass = int(labels[i])
        predclass = int(predictions[i])
        count[realclass, predclass] += 1.
    if norm_ax == 0:
        count = normalize(count, axis=1, norm='l1')
        count = normalize(count, axis=0, norm='l1')
    elif norm_ax == 1:
        count = normalize(count, axis=1, norm='l1')
    
    plt.clf()
    plt.imshow(count, cmap='autumn')
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')
    plt.colorbar()      
    fig = plt.gcf()
    ax  = plt.gca() 
    fig.set_size_inches(10, 10)
    for x in range(max(labels)+1):
        for y in range(max(labels)+1):
            ax.text(x, y, "{:.2f}".format(np.round(count[y,x], decimals=2)), horizontalalignment='center', verticalalignment='center')
    if do_return:
        return fig, ax
    else:
        return None







