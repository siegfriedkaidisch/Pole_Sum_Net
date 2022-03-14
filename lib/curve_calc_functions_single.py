# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Functions, that calculate the function values of different pole configurations on the real axis

"""
import numpy as np

from lib.pole_functions import complex_conjugate_pole_pair
    

def pole_curve_calc_single(pole_class, pole_params, grid_x):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    "_single" means, that this function deals with only 1 pole config
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=4 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
    
    returns: numpy.ndarray of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = np.reshape(grid_x, (-1))  
    pole_params = pole_params.transpose()
    
    if pole_class in [0,1]:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3])
    elif pole_class in [2,3,4]:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3]) + complex_conjugate_pole_pair(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7])
    elif pole_class in [5,6,7,8]:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3]) + complex_conjugate_pole_pair(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7]) + complex_conjugate_pole_pair(grid_x, pole_params[8], pole_params[9], pole_params[10], pole_params[11])
    return curve_pred


def pole_curve_calc_dens_single(pole_class, pole_params, grid_x):
    '''
    Calculate the real part of given pole configurations on a given grid

    "_single" means, that this function deals with only 1 pole config

    "_dens" means, that this function deals with pole configs, where the imaginary parts of real poles have been removed (Without '_dens' in the name these imaginary parts are kept in and are set to zero.) 
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: numpy.ndarray of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
    
    returns: numpy.ndarray of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = np.reshape(grid_x, (-1))  
    pole_params = pole_params.transpose()
    
    if pole_class == 0:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0)
    elif pole_class == 1:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3])
    elif pole_class == 2:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[2], pole_params[0]*0, pole_params[3], pole_params[0]*0)
    elif pole_class == 3:
        curve_pred = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[2], pole_params[3], pole_params[4], pole_params[5])
    elif pole_class == 4:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3]) + complex_conjugate_pole_pair(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7])
    elif pole_class == 5:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[2], pole_params[0]*0, pole_params[3], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[4], pole_params[0]*0, pole_params[5], pole_params[0]*0)
    elif pole_class == 6:
        curve_pred = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[2], pole_params[0]*0, pole_params[3], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7])
    elif pole_class == 7:
        curve_pred = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0) + complex_conjugate_pole_pair(grid_x, pole_params[2], pole_params[3], pole_params[4], pole_params[5]) + complex_conjugate_pole_pair(grid_x, pole_params[6], pole_params[7], pole_params[8], pole_params[9])
    elif pole_class == 8:
        curve_pred   = complex_conjugate_pole_pair(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3]) + complex_conjugate_pole_pair(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7]) + complex_conjugate_pole_pair(grid_x, pole_params[8], pole_params[9], pole_params[10], pole_params[11])
    return curve_pred



