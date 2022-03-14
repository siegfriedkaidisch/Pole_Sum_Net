#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:18:39 2020

@author: andreaskrassnigg, siegfriedkaidisch

Complex singularity data generation code: Generate data for the classifier

"""
import numpy as np
import os
from pathlib import Path
import shutil

from lib.get_params_functions        import get_train_params_dual

from lib.scipy_fit_functions_dual    import get_all_scipy_preds_dataprep_dual
from lib.curve_calc_functions_dual   import pole_curve_calc_dual, pole_curve_calc_dens_dual

from lib.scipy_fit_functions_single  import get_all_scipy_preds_dataprep_single
from lib.curve_calc_functions_single import pole_curve_calc_single, pole_curve_calc_dens_single

from lib.diverse_functions           import mse, drop_not_finite_rows
from lib.standardization_functions   import rm_std_data, std_data_new


def get_data_x_dual(data_y, grid_x, 
               re_max, re_min, im_max, im_min, 
               coeff_re_max, coeff_re_min, 
               coeff_im_max, coeff_im_min,
               method='lm', with_bounds=False):
    '''
    Generates the unstandardized network input from the pole curves (out_re)
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    data_y: numpy.ndarray of shape (n,) or (m,n), where m is the number of samples
        Function values at the gridpoints
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
        
    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method
   
    with_bounds: bool, default=False
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
           
    returns: two numpy.ndarrays of shapes (m,9+60) and (m,len(grid_x))
        data_x (network input) and the function values, not yet normalized
    '''
    data_y = np.atleast_2d(data_y)
    
    # Get Scipy predictions for each sample
    print('Getting SciPy predictions...')
    params_1r, params_1c, params_2r, params_1r1c, params_2c, \
    params_3r, params_2r1c, params_1r2c, params_3c = get_all_scipy_preds_dataprep_dual(grid_x=grid_x, data_y=data_y, 
                                                                                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                                                                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                                                                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                                                                  with_bounds=with_bounds,
                                                                                  method=method)

    # Calculate out_re for the different predicted pole configurations
    out_re_1r   = pole_curve_calc_dens_dual(pole_class=0, pole_params=params_1r,   grid_x=grid_x)
    out_re_1c   = pole_curve_calc_dens_dual(pole_class=1, pole_params=params_1c,   grid_x=grid_x)
    out_re_2r   = pole_curve_calc_dens_dual(pole_class=2, pole_params=params_2r,   grid_x=grid_x)
    out_re_1r1c = pole_curve_calc_dens_dual(pole_class=3, pole_params=params_1r1c, grid_x=grid_x)
    out_re_2c   = pole_curve_calc_dens_dual(pole_class=4, pole_params=params_2c,   grid_x=grid_x)
    out_re_3r   = pole_curve_calc_dens_dual(pole_class=5, pole_params=params_3r,   grid_x=grid_x)
    out_re_2r1c = pole_curve_calc_dens_dual(pole_class=6, pole_params=params_2r1c, grid_x=grid_x)
    out_re_1r2c = pole_curve_calc_dens_dual(pole_class=7, pole_params=params_1r2c, grid_x=grid_x)
    out_re_3c   = pole_curve_calc_dens_dual(pole_class=8, pole_params=params_3c,   grid_x=grid_x)
    
    '''
    Note: If SciPy fails and returns a nan, then pole_functions raises 'RuntimeWarning: invalid value encountered in true_divide'.
          We don't need to worry about this, since we just drop those samples.
    '''

    # Calculate the different MSEs
    mse_1r    = mse(data_y, out_re_1r,   ax=1).reshape(-1,1) 
    mse_1c    = mse(data_y, out_re_1c,   ax=1).reshape(-1,1) 
    mse_2r    = mse(data_y, out_re_2r,   ax=1).reshape(-1,1) 
    mse_1r1c  = mse(data_y, out_re_1r1c, ax=1).reshape(-1,1)
    mse_2c    = mse(data_y, out_re_2c,   ax=1).reshape(-1,1) 
    mse_3r    = mse(data_y, out_re_3r,   ax=1).reshape(-1,1) 
    mse_2r1c  = mse(data_y, out_re_2r1c, ax=1).reshape(-1,1) 
    mse_1r2c  = mse(data_y, out_re_1r2c, ax=1).reshape(-1,1) 
    mse_3c    = mse(data_y, out_re_3c,   ax=1).reshape(-1,1) 
    
    # Apply log10 to the MSEs to bring them to a similiar scale
    mse_1r   = np.log10(mse_1r)
    mse_1c   = np.log10(mse_1c)
    mse_2r   = np.log10(mse_2r)
    mse_1r1c = np.log10(mse_1r1c)
    mse_2c   = np.log10(mse_2c)
    mse_3r   = np.log10(mse_3r)
    mse_2r1c = np.log10(mse_2r1c)
    mse_1r2c = np.log10(mse_1r2c)
    mse_3c   = np.log10(mse_3c)
    
    # IF MSE=0 (perfect fit), then log(MSE)=-inf. Set to a finite value instead:
    mse_1r[mse_1r==-np.inf]     = np.min(mse_1r[np.isfinite(mse_1r)])
    mse_1c[mse_1c==-np.inf]     = np.min(mse_1c[np.isfinite(mse_1c)])
    mse_2r[mse_2r==-np.inf]     = np.min(mse_2r[np.isfinite(mse_2r)])
    mse_1r1c[mse_1r1c==-np.inf] = np.min(mse_1r1c[np.isfinite(mse_1r1c)])
    mse_2c[mse_2c==-np.inf]     = np.min(mse_2c[np.isfinite(mse_2c)])
    mse_3r[mse_3r==-np.inf]     = np.min(mse_3r[np.isfinite(mse_3r)])
    mse_2r1c[mse_2r1c==-np.inf] = np.min(mse_2r1c[np.isfinite(mse_2r1c)])
    mse_1r2c[mse_1r2c==-np.inf] = np.min(mse_1r2c[np.isfinite(mse_1r2c)])
    mse_3c[mse_3c==-np.inf]     = np.min(mse_3c[np.isfinite(mse_3c)])
    
    # Everything together then gives data_x that is used to train the classifier
    data_x = np.hstack((mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c))
    return data_x, data_y


def get_data_x_single(data_y, grid_x, 
               re_max, re_min, im_max, im_min, 
               coeff_re_max, coeff_re_min, 
               coeff_im_max, coeff_im_min,
               method='lm', with_bounds=False):
    '''
    Generates the unstandardized network input from the pole curves (out_re)
    
    "_single" means, that this function deals with only 1 pole config
    
    data_y: numpy.ndarray of shape (n,) or (m,n), where m is the number of samples
        Function values at the gridpoints
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
          
    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method

    with_bounds: bool, default=False
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
        
    returns: two numpy.ndarrays of shapes (m,9+60) and (m,len(grid_x))
        data_x (network input) and the function values, not yet normalized
    '''
    data_y = np.atleast_2d(data_y)
    
    # Get Scipy predictions for each sample
    print('Getting SciPy predictions...')
    params_1r, params_1c, params_2r, params_1r1c, params_2c, \
    params_3r, params_2r1c, params_1r2c, params_3c = get_all_scipy_preds_dataprep_single(grid_x=grid_x, data_y=data_y, 
                                                                                  re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                                                                  coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                                                                  coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                                                                  with_bounds=with_bounds,
                                                                                  method=method)

    # Calculate out_re for the different predicted pole configurations
    out_re_1r   = pole_curve_calc_dens_single(pole_class=0, pole_params=params_1r,   grid_x=grid_x)
    out_re_1c   = pole_curve_calc_dens_single(pole_class=1, pole_params=params_1c,   grid_x=grid_x)
    out_re_2r   = pole_curve_calc_dens_single(pole_class=2, pole_params=params_2r,   grid_x=grid_x)
    out_re_1r1c = pole_curve_calc_dens_single(pole_class=3, pole_params=params_1r1c, grid_x=grid_x)
    out_re_2c   = pole_curve_calc_dens_single(pole_class=4, pole_params=params_2c,   grid_x=grid_x)
    out_re_3r   = pole_curve_calc_dens_single(pole_class=5, pole_params=params_3r,   grid_x=grid_x)
    out_re_2r1c = pole_curve_calc_dens_single(pole_class=6, pole_params=params_2r1c, grid_x=grid_x)
    out_re_1r2c = pole_curve_calc_dens_single(pole_class=7, pole_params=params_1r2c, grid_x=grid_x)
    out_re_3c   = pole_curve_calc_dens_single(pole_class=8, pole_params=params_3c,   grid_x=grid_x)
    
    '''
    Note: If SciPy fails and returns a nan, then pole_functions raises 'RuntimeWarning: invalid value encountered in true_divide'.
          We don't need to worry about this, since we just drop those samples.
    '''

    # Calculate the different MSEs
    mse_1r    = mse(data_y, out_re_1r,   ax=1).reshape(-1,1) 
    mse_1c    = mse(data_y, out_re_1c,   ax=1).reshape(-1,1) 
    mse_2r    = mse(data_y, out_re_2r,   ax=1).reshape(-1,1) 
    mse_1r1c  = mse(data_y, out_re_1r1c, ax=1).reshape(-1,1)
    mse_2c    = mse(data_y, out_re_2c,   ax=1).reshape(-1,1) 
    mse_3r    = mse(data_y, out_re_3r,   ax=1).reshape(-1,1) 
    mse_2r1c  = mse(data_y, out_re_2r1c, ax=1).reshape(-1,1) 
    mse_1r2c  = mse(data_y, out_re_1r2c, ax=1).reshape(-1,1) 
    mse_3c    = mse(data_y, out_re_3c,   ax=1).reshape(-1,1) 
    
    # Apply log10 to the MSEs to bring them to a similiar scale
    mse_1r   = np.log10(mse_1r)
    mse_1c   = np.log10(mse_1c)
    mse_2r   = np.log10(mse_2r)
    mse_1r1c = np.log10(mse_1r1c)
    mse_2c   = np.log10(mse_2c)
    mse_3r   = np.log10(mse_3r)
    mse_2r1c = np.log10(mse_2r1c)
    mse_1r2c = np.log10(mse_1r2c)
    mse_3c   = np.log10(mse_3c)
    
    # IF MSE=0 (perfect fit), then log(MSE)=-inf. Set to a finite value instead:
    mse_1r[mse_1r==-np.inf]     = np.min(mse_1r[np.isfinite(mse_1r)])
    mse_1c[mse_1c==-np.inf]     = np.min(mse_1c[np.isfinite(mse_1c)])
    mse_2r[mse_2r==-np.inf]     = np.min(mse_2r[np.isfinite(mse_2r)])
    mse_1r1c[mse_1r1c==-np.inf] = np.min(mse_1r1c[np.isfinite(mse_1r1c)])
    mse_2c[mse_2c==-np.inf]     = np.min(mse_2c[np.isfinite(mse_2c)])
    mse_3r[mse_3r==-np.inf]     = np.min(mse_3r[np.isfinite(mse_3r)])
    mse_2r1c[mse_2r1c==-np.inf] = np.min(mse_2r1c[np.isfinite(mse_2r1c)])
    mse_1r2c[mse_1r2c==-np.inf] = np.min(mse_1r2c[np.isfinite(mse_1r2c)])
    mse_3c[mse_3c==-np.inf]     = np.min(mse_3c[np.isfinite(mse_3c)])
    
    # Everything together then gives data_x that is used to train the classifier
    data_x = np.hstack((mse_1r, mse_1c, mse_2r, mse_1r1c, mse_2c, mse_3r, mse_2r1c, mse_1r2c, mse_3c, params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c))
    return data_x, data_y


def create_training_data_classifier(length, grid_x, 
                                    re_max, re_min, im_max, im_min, 
                                    coeff_re_max, coeff_re_min, 
                                    coeff_im_max, coeff_im_min,
                                    data_dir, 
                                    method, with_bounds, 
                                    stage = 'creation', application_data = None):
    '''
    Creates training data for the NN classifier and saves it to the disk, or returns it, if stage=="application"
    
    Data is created by performing SciPy fits with different methods, calculating MSEs and concatenating everything
    
    length: int>0 
        The number of samples to be generated 
        
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
        
    data_dir: str
        Path to the folder, where data and standardization files shall be stored
        
    method: list of strings: 'trf', 'dogbox' or 'lm'
        The optimization methods (multiple possible, this is, why this is a list)

    with_bounds: list of bools
        Shall the Scipy fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
    
    stage: str: 'creation' or 'application'  
        'creation': Create new curves and data_x
        'application': Create data_x from curves passed as application_data
        
    application_data: numpy.ndarray of shape (n,) or (m,n)
        Data curves to create data_x for
    
    returns: None, if stage=='creation'
             numpy.ndarray of shape (m,n), if stage=='application'
    '''
    if length < 18:
        raise ValueError('length must be at least 18!')
    
    if stage == 'creation':
        # List of the pole classes
        pole_classes = [0,1,2,3,4,5,6,7,8]
        
        # calculate the number of samples per class
        length = np.floor(length/2) #due to pole swapping
        num    = int(np.floor(length/len(pole_classes)))
        
        # out_re will contain the real part of the pole curves; labels is the labels (0,1,2,..8)
        out_re = []
        labels_and_params = []
        for pole_class in pole_classes:
            # Get pole configurations of the current pole class and append them to out_re and labels
            params = get_train_params_dual(pole_class=pole_class, m=num,
                                      re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                      coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                      coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min)
            out_re.append(pole_curve_calc_dual(pole_class=pole_class, pole_params=params, grid_x=grid_x))
            
            # also add the exact parameters to the label, because we may use them later (e.g. for selecting certain samples)
            labels_and_params.append(np.hstack([np.ones([params.shape[0], 1])*pole_class, params]))
            
        # Convert lists to numpy arrays
        out_re = np.vstack(out_re)
        # add padding to the labels/params arrays
        max_coll = max( [arr.shape[1] for arr in labels_and_params] )
        for i in range(len(labels_and_params)):
            fillup = int(max_coll - labels_and_params[i].shape[1])
            labels_and_params[i] = np.hstack([labels_and_params[i], np.zeros([labels_and_params[i].shape[0], fillup])])
        labels_and_params = np.vstack(labels_and_params)
        print('Maximum number of samples to be created: ', len(labels_and_params)*2)
    elif stage == 'application':
        out_re            = np.atleast_2d(application_data)
        labels_and_params = np.zeros((out_re.shape[0],18+1)) # does not contain info. Just has to be created so the code below does not crash.
    else:
        raise ValueError('"stage" must be "creation" or "application"!')

    ###########################################################################
    # Get data_x
    for i in range(len(method)): #use different SciPy fitting methods
        print('method: ', method[i])
        print('with_bounds: ', with_bounds[i])
        
        ####################   Dual Joined Fitting   ##########################
        
        data_x_i, out_re = get_data_x_dual(data_y=out_re, grid_x=grid_x, 
                            re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                            coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                            coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                            with_bounds=with_bounds[i], 
                            method=method[i])  
        if i==0:
            data_x = data_x_i
        else:
            data_x = np.hstack([data_x, data_x_i])
        
        ## Get rid of infinities that occur if the sample couldn't be fitted
        data_x, out_re, labels_and_params = drop_not_finite_rows(
                                    data_x, out_re, labels_and_params)
        
        ####################   Dual Separate Fitting   ########################
        out_re1 = out_re[:,0:len(grid_x) ]
        out_re2 = out_re[:,  len(grid_x):]
        
        for out_re_i in [out_re1, out_re2]:
            data_x_i, _ = get_data_x_single(data_y=out_re_i, grid_x=grid_x, 
                                re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                with_bounds=with_bounds[i], 
                                method=method[i])  
            data_x = np.hstack([data_x, data_x_i])
        
        ## Get rid of infinities that occur if the sample couldn't be fitted
        data_x, out_re, labels_and_params = drop_not_finite_rows(
                                    data_x, out_re, labels_and_params)
        #######################################################################
    if len(method)==0:
        data_x = None
    
    ###########################################################################
    #Swap Poles around to get twice the data for free
    
    # Seperate labels and params array
    params = labels_and_params[:,1:]
    labels = labels_and_params[:,0].reshape(-1)
    
    #############   Swap Poles around to create more data   ###################
    # Swap data_x
    data_x_swapped = data_x.copy()
    for i in range(len(method)):
        # Swap Poles in Joined Fits
        data_x_swapped[:, np.array([ 9,10,11 ]) + i*237 ]                                                   = data_x_swapped[:, np.array([ 9,11,10 ]) + i*237 ]
        data_x_swapped[:, np.array([ 12,13,14,15,16,17 ]) + i*237 ]                                         = data_x_swapped[:, np.array([ 12,13,16,17,14,15 ]) + i*237 ]
        data_x_swapped[:, np.array([ 18,19,20,  21,22,23 ]) + i*237 ]                                       = data_x_swapped[:, np.array([ 18,20,19,  21,23,22 ]) + i*237 ]
        data_x_swapped[:, np.array([ 24,25,26,  27,28,29,30,31,32 ]) + i*237 ]                              = data_x_swapped[:, np.array([ 24,26,25,  27,28,31,32,29,30 ]) + i*237 ]
        data_x_swapped[:, np.array([ 33,34,35,36,37,38,  39,40,41,42,43,44 ]) + i*237 ]                     = data_x_swapped[:, np.array([ 33,34,37,38,35,36,  39,40,43,44,41,42 ]) + i*237 ]
        data_x_swapped[:, np.array([ 45,46,47,  48,49,50,  51,52,53 ]) + i*237 ]                            = data_x_swapped[:, np.array([ 45,47,46,  48,50,49,  51,53,52 ]) + i*237 ]
        data_x_swapped[:, np.array([ 54,55,56,  57,58,59,  60,61,62,63,64,65 ]) + i*237 ]                   = data_x_swapped[:, np.array([ 54,56,55,  57,59,58,  60,61,64,65,62,63 ]) + i*237 ]
        data_x_swapped[:, np.array([ 66,67,68,  69,70,71,72,73,74,  75,76,77,78,79,80 ]) + i*237 ]          = data_x_swapped[:, np.array([ 66,68,67,  69,70,73,74,71,72,  75,76,79,80,77,78 ]) + i*237 ]
        data_x_swapped[:, np.array([ 81,82,83,84,85,86,  87,88,89,90,91,92,  93,94,95,96,97,98 ]) + i*237 ] = data_x_swapped[:, np.array([ 81,82,85,86,83,84,  87,88,91,92,89,90,  93,94,97,98,95,96 ]) + i*237 ]
        # Swap the Separate Fits
        tmp                                         = data_x_swapped[:,  99 + i*237:167 + i*237 ]
        data_x_swapped[:,  99 + i*237:167 + i*237 ] = data_x_swapped[:, 168 + i*237:236 + i*237 ]
        data_x_swapped[:, 168 + i*237:236 + i*237 ] = tmp
    data_x = np.vstack([data_x, data_x_swapped])
        
    # Swap the curves
    out_re_swapped = out_re.copy()
    tmp                                = out_re_swapped[:, 0: len(grid_x) ]
    out_re_swapped[:, 0: len(grid_x) ] = out_re_swapped[:,  len(grid_x):  ]
    out_re_swapped[:,  len(grid_x):  ] = tmp
    out_re = np.vstack([out_re, out_re_swapped])
    
    # Add Swaped Poles to Params Array
    params_swapped = params.copy()
    params_swapped = params_swapped[:, [0,1,4,5,2,3,  6,7,10,11,8,9,  12,13,16,17,14,15]]
    params         = np.vstack([params,params_swapped])
    
    # Add Swaped Poles to Labels Array
    labels = np.hstack([labels,labels])
    
    ###########################################################################
    if stage == 'creation':
        # Standardize data_x
        data_x = std_data_new(data_x, with_mean=True, std_path=data_dir)
        
        # Save training data
        np.save(os.path.join(data_dir, 'pole_classifier_data_x.npy'), data_x)
        np.save(os.path.join(data_dir, 'pole_classifier_data_y.npy'), labels)
        np.save(os.path.join(data_dir, 'pole_classifier_curves.npy'), out_re)
        np.save(os.path.join(data_dir, 'pole_classifier_params.npy'), params)
        print("Successfully saved x data of shape ", np.shape(data_x))
        print("Successfully saved y data of shape ", np.shape(labels))
        print("Successfully saved curve data of shape ", np.shape(out_re))
        print("Successfully saved params data of shape ", np.shape(params))
        print('Samples per Class:')
        for i in range(9):
            print(str(i) + ': ', np.sum(labels==i))
        return
    elif stage == 'application':
        return data_x   #Note: returns two rows for each row in application_data, due to the pole swapping

    




    
#
