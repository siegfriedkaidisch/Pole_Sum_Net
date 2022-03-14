#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg

Complex singluarity data generation code: Generate data for the regressor

"""
import numpy as np
import os

from lib.get_params_functions      import get_train_params_dual
from lib.standardization_functions import std_data_new
from lib.curve_calc_functions_dual import pole_curve_calc_dens_dual 
from lib.pole_config_organize_dual import pole_config_organize_abs_dens_dual 
from lib.pole_config_organize_dual import remove_zero_imag_parts_dual


def create_training_data_regressor(length, pole_class, grid_x, data_dir,
                                   re_max, re_min, im_max, im_min, 
                                   coeff_re_max, coeff_re_min, 
                                   coeff_im_max, coeff_im_min):
    '''
    Creates training data for the NN regressor

    length: int>0
        The number of samples to be generated
        
    pole_class: int: 0-8
        The pole class
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored/ loaded from

    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
        
    returns: None or 2 numpy.ndarrays of shapes (length,n) and (length,k), where n is the number of gridpoints and k depends on the pole class
        see: mode
    '''
    # Generate pole configurations
    params = get_train_params_dual(pole_class=pole_class, m=length, 
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)

    # Drop the imaginary parts of the real poles
    params = remove_zero_imag_parts_dual(pole_class=pole_class, pole_params=params)

    # Calculate the pole curves
    out_re = pole_curve_calc_dens_dual(pole_class=pole_class, pole_params=params, grid_x=grid_x)
    # Organize pole configurations
    params = pole_config_organize_abs_dens_dual(pole_class=pole_class, pole_params=params)

    # Standardize Inputs
    out_re = std_data_new(data=out_re, with_mean=False, name_var="variances.npy", std_path=data_dir) 
    #Standardize Outputs
    params = std_data_new(data=params, with_mean=True, name_var="variances_params.npy", 
                              name_mean="means_params.npy", std_path=data_dir)
    # Save training data
    np.save(os.path.join(data_dir, 'various_poles_data_regressor_x.npy'), out_re)
    np.save(os.path.join(data_dir, 'various_poles_data_regressor_y.npy'), params)
    print("Successfully saved x data of shape ", np.shape(out_re))
    print("Successfully saved y data of shape ", np.shape(params))
    return None






#
