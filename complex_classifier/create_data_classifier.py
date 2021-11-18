#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Complex singularity data generation code: Generate data for the classifier

"""
import time

from lib.training_data_generation_classifier import create_training_data_classifier

from parameters import n_examples_classifier, standard_re, data_dir_classifier
from parameters import xtol_classifier
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import p0_classifier, method_classifier, maxfev_classifier, num_tries_classifier, with_bounds_classifier

if __name__ == '__main__':
    time1 = time.time()
    create_training_data_classifier(length=n_examples_classifier, grid_x=standard_re,
                                    re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                    coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                    coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                    with_bounds=with_bounds_classifier, data_dir=data_dir_classifier, 
                                    p0=p0_classifier, method=method_classifier, 
                                    maxfev=maxfev_classifier, num_tries=num_tries_classifier, 
                                    xtol=xtol_classifier)
    print(time.time() - time1)





