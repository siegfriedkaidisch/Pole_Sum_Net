#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Combine two classifier datasets into one

"""
import numpy as np
import os

from lib.standardization_functions import rm_std_data, std_data_new

from parameters import data_dir_classifier


def combine_classifier_datasets(with_mean=True, data_dir=data_dir_classifier):
    '''
    Combine two classifier datasets into one and saves the new dataset to the disk (to data_dir)
    
    with_mean: bool
        Shall the mean of each feature be shifted during the standardization?
    
    data_dir: str
        Path to the folder containing all necessary files. 
        Must have the following folder structure and file content:
            
        data_dir/A/various_poles_data_classifier_x.npy
        
        data_dir/A/various_poles_data_classifier_y.
        
        data_dir/A/various_poles_data_classifier_params.npy
        
        data_dir/A/variances.npy
        
        data_dir/A/means.npy
        
        data_dir/B/various_poles_data_classifier_x.npy
        
        data_dir/B/various_poles_data_classifier_y.npy
        
        data_dir/B/various_poles_data_classifier_params.npy
        
        data_dir/B/variances.npy
        
        data_dir/B/means.npy
        
    returns: None
    '''
    ##############################################################################
    ##########################   Dataset A    ####################################
    ##############################################################################
    data_dir_A = data_dir + 'A/'
    
    # Import the data, generated with create_data_classifier
    data_x_A = np.load(os.path.join(data_dir_A, "various_poles_data_classifier_x.npy"), allow_pickle=True).astype('float32')
    labels_A = np.load(os.path.join(data_dir_A, "various_poles_data_classifier_y.npy"), allow_pickle=True).astype('int64').reshape((-1,1))
    params_A = np.load(os.path.join(data_dir_A, "various_poles_data_classifier_params.npy"), allow_pickle=True).astype('float32')
    print("Successfully loaded x data of shape ", np.shape(data_x_A))
    print("Successfully loaded y data of shape ", np.shape(labels_A))
    print("Successfully loaded params data of shape ", np.shape(params_A))

    # Remove standardization from data_x
    data_x_A = rm_std_data(data=data_x_A, with_mean=with_mean, std_path=data_dir_A, name_var="variances.npy", name_mean="means.npy")
    
    ##############################################################################
    ##########################   Dataset B    ####################################
    ##############################################################################
    data_dir_B = data_dir + 'B/'
    
    # Import the data, generated with create_data_classifier
    data_x_B = np.load(os.path.join(data_dir_B, "various_poles_data_classifier_x.npy"), allow_pickle=True).astype('float32')
    labels_B = np.load(os.path.join(data_dir_B, "various_poles_data_classifier_y.npy"), allow_pickle=True).astype('int64').reshape((-1,1))
    params_B = np.load(os.path.join(data_dir_B, "various_poles_data_classifier_params.npy"), allow_pickle=True).astype('float32')
    print("Successfully loaded x data of shape ", np.shape(data_x_B))
    print("Successfully loaded y data of shape ", np.shape(labels_B))
    print("Successfully loaded params data of shape ", np.shape(params_B))

    # Remove standardization from data_x
    data_x_B = rm_std_data(data=data_x_B, with_mean=with_mean, std_path=data_dir_B, name_var="variances.npy", name_mean="means.npy")
    
    ##############################################################################
    ##########################   Combine A and B    ##############################
    ##############################################################################
    #Combine the arrays
    data_x = np.vstack([data_x_A, data_x_B])
    labels = np.vstack([labels_A, labels_B])
    params = np.vstack([params_A, params_B])

    # Standardize Inputs
    data_x = std_data_new(data_x, with_mean=with_mean, std_path=data_dir)
    
    # Save training data
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_x.npy'), data_x)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_y.npy'), labels)
    np.save(os.path.join(data_dir, 'various_poles_data_classifier_params.npy'), params)
    print("Successfully saved x data of shape ", np.shape(data_x))
    print("Successfully saved y data of shape ", np.shape(labels))
    print("Successfully saved params data of shape ", np.shape(params)) 
    
    return None




##############################################################################
##########################   Execution   #####################################
##############################################################################


if __name__ == '__main__':
   
    combine_classifier_datasets(with_mean=True, data_dir=data_dir_classifier)


###############################################################################
###############################################################################
###############################################################################


#
