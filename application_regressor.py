#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Example: Apply SciPy curv_fit, regressor, NN regressor to data 

"""
import numpy as np

from lib.application_regressor_functions import get_regressor_pred

from parameters import dir_regressor


##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    # Get Data 
    from test_data import my_sigma_S, my_sigma_V
    data_y = np.hstack([my_sigma_S, my_sigma_V])


    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Regressor prediction:')
    params_pred_ANN = get_regressor_pred(data_y=data_y, model_path=dir_regressor)
    params_pred_ANN = np.mean(params_pred_ANN, axis=(0))
    print(params_pred_ANN)



###############################################################################
###############################################################################
###############################################################################

                        

#
