# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:18:44 2021

@author: siegfriedkaidisch

Functions to get predictions from NN regressor checkpoints

"""
import numpy as np
import torch
import os

from lib.standardization_functions import std_data, rm_std_data
from lib.pole_regressor import Pole_Regressor
    
    
def get_regressor_pred(data_y, model_path):
    '''
    Get predictions from a trained Pole Regressor
    
    data_y: ndarray of shape (n,) or (m,n), where m is the number of samples
        Y-values/Curve, whose parameters are to be found; multiple (m) at the same time are possible
    
    model_path: str
        Path to folder containing all necessary files:
            There must be a subdirectory called 'models' containing atleast one PL checkpoint of the Regressor.
            
            There must additionally be a subdirectory called 'data' containing the standardization files called 
            "variances.npy" for the inputs and "variances_params.npy", "means_params.npy" for the outputs.
    
    returns: ndarray of shape (l,m,k), where l is the number of trained models
        The predicted parameters are returned
    '''
    # Standardize input
    data_y = np.atleast_2d(data_y)
    data_y = std_data(data=data_y, std_path= os.path.join(model_path, 'data/'), 
                                with_mean=False, name_var="variances.npy")
    
    # Get prediction from Regressors
    params_pred = []
    for filename in os.listdir(os.path.join(model_path, 'models/')):
        model = Pole_Regressor.load_from_checkpoint(os.path.join(model_path, 'models/', filename))
        model.eval()
        pred = model(torch.from_numpy(data_y.astype('float32')))
        pred = pred.detach().numpy()
        # Remove standardization from output
        pred = rm_std_data(data=pred, std_path= os.path.join(model_path, 'data/'), 
                                   with_mean=True, name_var="variances_params.npy", name_mean="means_params.npy")
        params_pred.append( pred )
        del model
    params_pred = np.array(params_pred)
    return params_pred


def get_all_regressor_preds(data_y, model_path, regressor_subdirs):
    '''
    Get predictions from trained Pole Regressors for all 9 different Pole Configurations
    
    data_y: ndarray of shape (n,) or (m,n), where m is the number of samples
        Y-values/Curve, whose parameters are to be found; multiple (m) at the same time are possible
    
    model_path: str
        Path to folder containing all necessary files:
            Must contain a subdirectory for each of the 9 Pole Configurations: 
                '0-1r', '1-1c', '2-2r', '3-1r1c', '4-2c', '5-3r', '6-2r1c', '7-1r2c', '8-3c'
            
            Inside each of these subdirectories:
                There must be a subsubdirectory called 'models' containing atleast one PL checkpoint of the Regressor.
                
                There must additionally be a subsubdirectory called 'data' containing the standardization files called 
                "variances.npy" for the inputs and "variances_params.npy", "means_params.npy" for the outputs.
    
    regressor_subdirs: list of str        
        list of the names of the subdirs of the individual regressors
    
    returns: list of 9 ndarrays of shapes (m, k_i), i=0...8
        Optimized parameters of the different Pole Configurations/Classes
    '''
    preds = []
    for subdir in regressor_subdirs:
        pred = get_regressor_pred(data_y=data_y, model_path=os.path.join(model_path, subdir + '/'))
        pred = np.mean(pred, axis=0) # average over all models in /models
        preds.append(pred)
    return preds


