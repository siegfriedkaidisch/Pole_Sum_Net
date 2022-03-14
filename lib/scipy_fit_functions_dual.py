# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:09:38 2021

@author: siegfriedkaidisch

Functions, that use SciPy's curve_fit to get pole parameters

"""
import numpy as np
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

from lib.pole_objective_functions_dual import objective_1r_dual, objective_1c_dual
from lib.pole_objective_functions_dual import objective_2r_dual, objective_1r1c_dual, objective_2c_dual
from lib.pole_objective_functions_dual import objective_3r_dual, objective_2r1c_dual, objective_1r2c_dual, objective_3c_dual
from lib.pole_objective_functions_dual import objective_1r_jac_dual, objective_1c_jac_dual
from lib.pole_objective_functions_dual import objective_2r_jac_dual, objective_1r1c_jac_dual, objective_2c_jac_dual
from lib.pole_objective_functions_dual import objective_3r_jac_dual, objective_2r1c_jac_dual, objective_1r2c_jac_dual, objective_3c_jac_dual
from lib.pole_config_organize_dual     import pole_config_organize_abs_dens_dual as pole_config_organize
from lib.pole_config_organize_dual     import check_inside_bounds_dens_dual


def get_scipy_pred_dual(pole_class, grid_x, data_y, 
                   re_max, re_min, im_max, im_min, 
                   coeff_re_max, coeff_re_min, 
                   coeff_im_max, coeff_im_min,
                   method='lm', with_bounds=False
                   ):
    '''
    Uses Scipy curve_fit to fit different pole classes onto single (!) data sample
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    pole_class: int = 0-8 
        The class of the pole configuration to be found
    
    grid_x, data_y: numpy.ndarray of shape (n,) or (1,n)
        Points to be used for fitting
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method

    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
     
    returns: numpy.ndarray of shape (k,)
        Optimized parameters of the chosen pole class or nans if the fit failed
    ''' 
    xtol       = 1e-8
    num_tries  = 10
    if method == 'lm':
        maxfev = [10000, 100000, 1000000, 10000000]
    else:
        maxfev = [1000, 10000, 100000, 1000000, 10000000] 
    
    grid_x = np.reshape(grid_x,(-1))
    data_y = np.reshape(data_y,(-1))
    
    drop_outside_bounds = with_bounds
    if method == 'lm':
        with_bounds = False
             
    def get_p0(lower, upper):
        return np.random.uniform(np.array(lower), np.array(upper))

    for maxfev_i in maxfev:                
        for num_try in range(num_tries): #retry fit num_tries times (with different random p0)
            try:
                if pole_class == 0:
                    lower = [re_min, -coeff_re_max, -coeff_re_max]
                    upper = [re_max, coeff_re_max, coeff_re_max] 
                    p0_new = get_p0(lower, upper)         
                    params_tmp, _ = curve_fit(objective_1r_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_1r_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_1r_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_1r_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 1:
                    lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max]
                    upper = [re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max]   
                    p0_new = get_p0(lower, upper)          
                    params_tmp, _ = curve_fit(objective_1c_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_1c_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_1c_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_1c_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 2:
                    lower = [re_min, -coeff_re_max, -coeff_re_max, re_min, -coeff_re_max, -coeff_re_max]
                    upper = [re_max, coeff_re_max, coeff_re_max, re_max, coeff_re_max, coeff_re_max]   
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_2r_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_2r_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_2r_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_2r_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 3:
                    lower = [re_min, -coeff_re_max, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max]
                    upper = [re_max, coeff_re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max]
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_1r1c_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_1r1c_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_1r1c_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_1r1c_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 4:
                    lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max]
                    upper = [re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max]
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_2c_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_2c_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_2c_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_2c_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 5:
                    lower = [re_min, -coeff_re_max, -coeff_re_max, re_min, -coeff_re_max, -coeff_re_max, re_min, -coeff_re_max, -coeff_re_max]
                    upper = [re_max, coeff_re_max, coeff_re_max, re_max, coeff_re_max, coeff_re_max, re_max, coeff_re_max, coeff_re_max]
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_3r_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_3r_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_3r_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_3r_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 6:
                    lower = [re_min, -coeff_re_max, -coeff_re_max, re_min, -coeff_re_max, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max]
                    upper = [re_max, coeff_re_max, coeff_re_max, re_max, coeff_re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max]
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_2r1c_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_2r1c_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_2r1c_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_2r1c_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 7:
                    lower = [re_min, -coeff_re_max, -coeff_re_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max]
                    upper = [re_max, coeff_re_max, coeff_re_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max]
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_1r2c_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_1r2c_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_1r2c_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_1r2c_jac_dual, xtol=xtol, method=method)
    
                elif pole_class == 8:
                    lower = [re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max, re_min, im_min, -coeff_re_max, -coeff_im_max, -coeff_re_max, -coeff_im_max]
                    upper = [re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max, re_max, im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max]
                    p0_new = get_p0(lower, upper)
                    params_tmp, _ = curve_fit(objective_3c_dual, grid_x, data_y, maxfev=maxfev_i, bounds=(lower, upper), p0=p0_new, jac=objective_3c_jac_dual, xtol=xtol, method=method) if with_bounds else \
                                  curve_fit(objective_3c_dual, grid_x, data_y, maxfev=maxfev_i, p0=p0_new, jac=objective_3c_jac_dual, xtol=xtol, method=method)
            
                if drop_outside_bounds: # Check if the fitted parameters are inside bounds (only relevant for 'lm')
                    in_bounds = check_inside_bounds_dens_dual(pole_class, params_tmp, 
                                            re_max, re_min, im_max, im_min, 
                                            coeff_re_max, coeff_re_min, 
                                            coeff_im_max, coeff_im_min)[0]
                    if not in_bounds:
                        params_tmp = np.array([np.nan for i in range(len(lower))])
            
            except:
                params_tmp = np.array([np.nan for i in range(len(lower))])
                        
            if ~np.isnan(params_tmp[0]):    # If the fit worked, break the retry loop
                break
        if ~np.isnan(params_tmp[0]):    # If the fit worked, break the retry loop
            break
    
    if np.isnan(params_tmp[0]):   
        if method == 'lm' and drop_outside_bounds:
            print('Fit failed... For method="lm" and with_bounds=True, this is probably since the fit did not converge inside the bounds.')
        else:
            print('Fit failed! Try a higher value for "maxfev", "xtol" or "num_tries".')      
    params_tmp = pole_config_organize(pole_class=pole_class, pole_params=params_tmp.reshape(1,-1)).reshape(-1) 
    return params_tmp


def get_all_scipy_preds_dual(grid_x, data_y, 
                        re_max, re_min, im_max, im_min, 
                        coeff_re_max, coeff_re_min, 
                        coeff_im_max, coeff_im_min,
                        method='lm', with_bounds=False):
    '''
    Uses Scipy curve_fit to fit all 9 different pole classes onto a single data sample
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    grid_x, data_y: numpy.ndarray of shape (n,) or (1,n)
        Points to be used for fitting
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True
    
    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method

    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
   
    returns: list of 9 numpy.ndarrays of shapes (k_i,), i=0...8
        Optimized parameters of the different pole classes
    '''
    params = []
    for i in range(9):
        params_tmp = get_scipy_pred_dual(pole_class=i, grid_x=grid_x, data_y=data_y, 
                                    re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                    coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                    coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                    with_bounds=with_bounds,
                                    method=method)
        params.append(params_tmp)
        if np.isnan(params_tmp[0]): # if one fit fails, the sample will be dropped, so break to not waste time
            params = [np.array([np.nan for i in range(j)]) for j in [3,6, 6,9,12, 9,12,15,18]]
            break    
    return params


def get_all_scipy_preds_dataprep_dual(grid_x, data_y, 
                                 re_max, re_min, im_max, im_min, 
                                 coeff_re_max, coeff_re_min, 
                                 coeff_im_max, coeff_im_min,
                                 method='lm', with_bounds=False):
    '''
    Uses Scipy curve_fit to fit all 9 different pole classes onto multiple data samples for creating data to train a NN. 
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints
        
    data_y: numpy.ndarray of shape (n,) or (m,n), where m is the number of samples
        Function values to be fitted
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Define a box. Parameter configurations are searched in this box if with_bounds=True

    method: str = 'trf', 'dogbox' or 'lm', default='lm'
        The optimization method

    with_bounds: bool, default=False
        Shall the fit's parameters be contrained by bounds determined by coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min, re_min, re_max, im_min, im_max?
            
    returns: 9 numpy.ndarrays of shapes (m,k_i) for i=0...8, where m is the number of samples
        optimized parameters (nans if the fit failed)
    '''
    grid_x = grid_x.reshape(-1)
    data_y = np.atleast_2d(data_y)

    def get_all_scipy_preds_tmp(data_y_fun):
        return get_all_scipy_preds_dual(grid_x=grid_x, data_y=data_y_fun, 
                                   re_max=re_max, re_min=re_min, im_max=im_max, im_min=im_min, 
                                   coeff_re_max=coeff_re_max, coeff_re_min=coeff_re_min, 
                                   coeff_im_max=coeff_im_max, coeff_im_min=coeff_im_min,
                                   with_bounds=with_bounds,
                                   method=method)
    
    params_tmp = Parallel(n_jobs=-1, backend="loky", verbose=10)(
                 map(delayed(get_all_scipy_preds_tmp), list(data_y)))
   
    params_1r   = [tmp[0] for tmp in params_tmp]
    params_1c   = [tmp[1] for tmp in params_tmp]
    params_2r   = [tmp[2] for tmp in params_tmp]
    params_1r1c = [tmp[3] for tmp in params_tmp]
    params_2c   = [tmp[4] for tmp in params_tmp]
    params_3r   = [tmp[5] for tmp in params_tmp]
    params_2r1c = [tmp[6] for tmp in params_tmp]
    params_1r2c = [tmp[7] for tmp in params_tmp]
    params_3c   = [tmp[8] for tmp in params_tmp]
 
    params_1r   = np.array(params_1r)
    params_1c   = np.array(params_1c)
    params_2r   = np.array(params_2r)
    params_1r1c = np.array(params_1r1c)
    params_2c   = np.array(params_2c)
    params_3r   = np.array(params_3r)
    params_2r1c = np.array(params_2r1c)
    params_1r2c = np.array(params_1r2c)
    params_3c   = np.array(params_3c)

    return params_1r, params_1c, params_2r, params_1r1c, params_2c, params_3r, params_2r1c, params_1r2c, params_3c













