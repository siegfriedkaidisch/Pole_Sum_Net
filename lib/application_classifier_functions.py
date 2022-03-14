# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Functions needed to apply the finished classifier to real data

"""
import numpy as np
import os
import torch
import scipy

from lib.standardization_functions import std_data
from lib.pole_classifier import Pole_Classifier
from lib.training_data_generation_classifier import create_training_data_classifier


def get_classifier_preds(grid_x, data_y, 
                         re_max, re_min, im_max, im_min, 
                         coeff_re_max, coeff_re_min, 
                         coeff_im_max, coeff_im_min,
                         do_std, model_path, 
                         with_bounds, method):
    '''
    Get predictions from trained Pole Classifier(s) 
    
    grid_x: ndarray of shape (n,) or (1,n), where n is the number of gridpoints
        Gridpoints
    
    data_y: ndarray of shape (n,) or (m,n), where n is the number of gridpoints
        Function Values

    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
    do_std: bool
        Shall the data be standardized to the standardization, that was used to train the Classifier?
    
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Classifier.
            
            If do_std=True, there must additionally be a subdirectory called 'data' containing the standardization files called "variances.npy" and "means.npy".
    
    with_bounds: bool
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?

    method: str = 'trf', 'dogbox' or 'lm'
        The optimization method

    returns: list of lists containing: int, int, ndarray of shape (l,), where l is twice the number of checkpoints in the "models" subdirectory, ndarray of shape (n,)
        Hard Averaging, Soft Averaging, Array with Predictions from the individual Checkpoints (two per checkpoint due to pole swap), the corresponding data curve
    '''
    data_y = np.atleast_2d(data_y)
    
    # Data Preparation
    # Get data_x
    data_x = create_training_data_classifier(length=None, grid_x=grid_x, 
                                        re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                        coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                        coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                        data_dir=None, 
                                        method=method, with_bounds=with_bounds, 
                                        stage = 'application', application_data = data_y)
    # NOTE: from each curve in data_y, we get two lines in data_x, due to the pole swapping in create_training_data_classifier
    
    # Apply standardization
    if do_std:
        data_x = std_data(data=data_x, std_path=os.path.join(model_path, 'data/'), with_mean=True)
    
    # Get predictions from Classifiers
    class_pred = []
    for filename in os.listdir(os.path.join(model_path, 'models/')):
        model = Pole_Classifier.load_from_checkpoint(os.path.join(model_path, 'models/', filename))
        model.eval()
        pred = model(torch.from_numpy(data_x.astype('float32')))
        class_pred.append( pred.detach().numpy() )
        del model
    class_pred = np.array(class_pred)
    class_pred = np.swapaxes(class_pred,0,1)
    
    num_curves = data_y.shape[0]
    preds = []
    for i in range(num_curves): # run over data curves
        # Combine the different predictions (from pole swapping) of each data curve into one
        class_pred_i = np.concatenate([class_pred[i,:,:], class_pred[i+num_curves]],axis=0)
        class_pred_i = np.expand_dims(class_pred_i,0)
        # Get the individual predictions
        individual_preds = np.argmax(class_pred_i, axis=2).reshape(-1)
    
        # Hard Averaging - Take most frequent prediction (=mode)
        class_pred_hard = np.argmax(class_pred_i, axis=2)
        class_pred_hard = scipy.stats.mode(class_pred_hard, axis=1)[0].item() 
        
        # Soft Averaging - sum over probabilities and the take argmax
        class_pred_soft = np.sum(class_pred_i, axis=1)
        class_pred_soft = np.argmax(class_pred_soft, axis=1).reshape((-1,1)).item()
        
        preds.append({'Soft_Average_Pred': class_pred_soft, 
                      'Hard_Average_Pred': class_pred_hard, 
                      'Individual_Preds': individual_preds, 
                      'Data_Curve': data_y[i]})
    return preds

















