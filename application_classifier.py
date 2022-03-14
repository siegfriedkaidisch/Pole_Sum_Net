#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Example: Apply SciPy curv_fit, classifier, NN regressor to data 

"""
import numpy as np

from lib.application_classifier_functions import get_classifier_preds

from parameters import dir_classifier
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import method_classifier, with_bounds_classifier
from parameters import standard_re as grid_x


##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    # Get Data 
    from test_data import my_sigma_S, my_sigma_V
    data_y = np.hstack([my_sigma_S, my_sigma_V])


    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Classifier prediction:')
    class_pred      = get_classifier_preds(grid_x=grid_x, data_y=data_y,
                                           re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                           coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                           coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                           do_std=True, model_path=dir_classifier,
                                           with_bounds=with_bounds_classifier, 
                                           method=method_classifier)
    print(class_pred)



###############################################################################
###############################################################################
###############################################################################

                        

#
