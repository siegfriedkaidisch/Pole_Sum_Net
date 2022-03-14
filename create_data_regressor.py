#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg

Complex singluarity data generation code: Generate data for the regressor

"""
from lib.training_data_generation_regressor import create_training_data_regressor

from parameters import class_regressor, data_dir_regressor
from parameters import standard_re
from parameters import n_examples_regressor
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min


if __name__ == '__main__':

    create_training_data_regressor(length=n_examples_regressor, pole_class=class_regressor, 
                         grid_x=standard_re, data_dir=data_dir_regressor,
                         re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                         coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                         coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)









#
