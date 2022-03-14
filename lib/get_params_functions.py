# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:06:39 2021

@author: siegfriedkaidisch

Functions, that generate pole configurations

"""
import numpy as np
import sys


def get_params_dual(m, typ: str,
               re_max, re_min, im_max, im_min, 
               coeff_re_max, coeff_re_min, 
               coeff_im_max, coeff_im_min):
    '''
    Generates parameters of two real or cc pole pairs at the same position, but with different coeffs. 
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    WARNING: Real poles are treated as cc pole pairs with zero imaginary parts. 
    Thus, in the end, you have to multiply coeff_re by 2.
    
    m: int>0
        The number of configurations to be created
        
    typ: str: 'r' or 'c'
        Shall the generated poles be real or cc pairs?
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations outside this box are dropped
        
    returns: numpy.ndarray of shape (m, 6)
        The generated pole configurations (pos_re, pos_im, coeff_re_1, coeff_im_1, coeff_re_2, coeff_im_2)
    '''
    
    if typ == 'r':
         part_re  = np.random.uniform(re_min, re_max, size=(m))
         
         part_im  = np.zeros((m))
         
         # Get Coeffs from +/- [coeff_min, coeff_max]
         sign = np.random.choice([-1,1], size=(m))
         tmp_min = np.min(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
         tmp_max = np.max(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
         coeff_re1 = np.random.uniform(tmp_min, tmp_max, size=(m)) 
         coeff_im1 = np.zeros((m))
         
         sign = np.random.choice([-1,1], size=(m))
         tmp_min = np.min(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
         tmp_max = np.max(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
         coeff_re2 = np.random.uniform(tmp_min, tmp_max, size=(m)) 
         coeff_im2 = np.zeros((m))
         
    elif typ == 'c':
        part_re  = np.random.uniform(re_min, re_max, size=(m))
        
        part_im  = np.random.uniform(im_min, im_max, size=(m))
        
        # Get Coeffs from +/- [coeff_min, coeff_max]
        sign = np.random.choice([-1,1], size=(m))
        tmp_min = np.min(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
        tmp_max = np.max(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
        coeff_re1 = np.random.uniform(tmp_min, tmp_max, size=(m)) 
        sign = np.random.choice([-1,1], size=(m))
        tmp_min = np.min(np.array([sign*coeff_im_min, sign*coeff_im_max]), axis = 0)
        tmp_max = np.max(np.array([sign*coeff_im_min, sign*coeff_im_max]), axis = 0)
        coeff_im1 = np.random.uniform(tmp_min, tmp_max, size=(m)) 
        
        sign = np.random.choice([-1,1], size=(m))
        tmp_min = np.min(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
        tmp_max = np.max(np.array([sign*coeff_re_min, sign*coeff_re_max]), axis = 0)
        coeff_re2 = np.random.uniform(tmp_min, tmp_max, size=(m)) 
        sign = np.random.choice([-1,1], size=(m))
        tmp_min = np.min(np.array([sign*coeff_im_min, sign*coeff_im_max]), axis = 0)
        tmp_max = np.max(np.array([sign*coeff_im_min, sign*coeff_im_max]), axis = 0)
        coeff_im2 = np.random.uniform(tmp_min, tmp_max, size=(m)) 
        
    else:
        sys.exit("Undefined type.")
        
    return (np.array([part_re, part_im, coeff_re1, coeff_im1, coeff_re2, coeff_im2])).transpose()


def get_train_params_dual(pole_class:int, m:int,
                     re_max, re_min, im_max, im_min, 
                     coeff_re_max, coeff_re_min, 
                     coeff_im_max, coeff_im_min):
    '''
    For a given pole class, return m parameter configurations
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    pole_class: int: 0-8
        The pole class
        
    m: int
        The number of configurations to be generated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations outside this box are dropped
        
    returns: numpy.ndarray of shape (m,k), where k is determined by the pole Class and m is the number of samples
        The generated configurations
    '''
    if pole_class == 0:
        # a single pole on the real axis
        params_1 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = params_1
    elif pole_class == 1:
        # a complex conjugated pole pair
        params_1 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = params_1
    elif pole_class == 2:
        # two real poles
        params_1 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min) 
        params_2 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = np.hstack([params_1, params_2])
    elif pole_class == 3:
        # a real pole and a cc pole pair
        params_1 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min) 
        params_2 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = np.hstack([params_1, params_2])
    elif pole_class == 4:
        # two cc pole pairs
        params_1 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)  
        params_2 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)  
        params  = np.hstack([params_1, params_2])
    elif pole_class == 5:
        # three real poles
        params_1 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_2 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_3 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = np.hstack([params_1, params_2, params_3])
    elif pole_class == 6:
        # two real poles and one cc pole pair
        params_1 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_2 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_3 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = np.hstack([params_1, params_2, params_3])
    elif pole_class == 7:
        # one real pole and two cc pole pairs
        params_1 = get_params_dual(m, 'r',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_2 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_3 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = np.hstack([params_1, params_2, params_3])
    elif pole_class == 8:
        # three cc pole pairs
        params_1 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_2 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params_3 = get_params_dual(m, 'c',
                              re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                              coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                              coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
        params  = np.hstack([params_1, params_2, params_3])
    else:
        # should not happen
        sys.exit("Undefined label.")
    return params



