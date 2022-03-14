#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Example: Apply SciPy curv_fit, regressor, NN regressor to data 

"""
import numpy as np
import contextlib

from lib.scipy_fit_functions_dual import get_scipy_pred_dual

from parameters import standard_re
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min


##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    # Get Data 
    from test_data import my_sigma_S, my_sigma_V
    data_y = np.hstack([my_sigma_S, my_sigma_V])

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('SciPy predictions (method=lm, with_bounds=True):')
    for pole_class in range(9):
        with contextlib.redirect_stdout(None):
            params_pred_lm = get_scipy_pred_dual(pole_class=pole_class, grid_x=standard_re, data_y=data_y, 
                               re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                               coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                               coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                               method='lm', with_bounds=True
                               )
        print('Pole Class ' + str(pole_class) + ': ')
        print(params_pred_lm)



###############################################################################
###############################################################################
###############################################################################

                        

#
