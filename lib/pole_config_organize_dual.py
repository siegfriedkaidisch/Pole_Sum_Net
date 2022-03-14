# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:09:38 2021

@author: siegfriedkaidisch

Functions, that organize pole configurations

"""
import numpy as np
import torch
import sys


def pole_config_organize_abs_dens_dual(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Parameters of real poles are always kept at the front
        
        Poles are ordered by Re(pole position)**2 + Im(pole position)**2 from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    "_dens" means, that this function deals with pole configs, where the imaginary parts of real poles have been removed (Without '_dens' in the name these imaginary parts are kept in and are set to zero.) 
    
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=3 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0:
        None
        
    elif pole_class == 1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        
    elif pole_class == 2:
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:3]
        params2 = pole_params[:, 3:6]
        val1    = params1[:,0]**2 
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [3,4,5,  0,1,2]]
        
    elif pole_class == 3:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,4]    <  0
        pole_params[indices,4]       *= -1
        pole_params[indices,6]       *= -1
        pole_params[indices,8]       *= -1
        
    elif pole_class == 4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 6:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,9,10,11,  0,1,2,3,4,5]]
        
    elif pole_class == 5:
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:3]
        params2 = pole_params[:, 3:6]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [3,4,5,  0,1,2,  6,7,8]]

        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:3]
        params2 = pole_params[:, 6:9]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,  3,4,5,  0,1,2]]
        
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 3:6]
        params2 = pole_params[:, 6:9]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,  6,7,8,  3,4,5]]
        
    elif pole_class == 6:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:3]
        params2 = pole_params[:, 3:6]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [3,4,5,  0,1,2,  6,7,8,9,10,11]]
        
    elif pole_class == 7:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,4]    <  0
        pole_params[indices,4]       *= -1
        pole_params[indices,6]       *= -1
        pole_params[indices,8]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,10]    <  0
        pole_params[indices,10]       *= -1
        pole_params[indices,12]       *= -1
        pole_params[indices,14]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 3:9]
        params2 = pole_params[:, 9:15]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,  9,10,11,12,13,14,  3,4,5,6,7,8]]
        
    elif pole_class == 8:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]      *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,13]    <  0
        pole_params[indices,13]       *= -1
        pole_params[indices,15]       *= -1
        pole_params[indices,17]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 6:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,9,10,11,  0,1,2,3,4,5,  12,13,14,15,16,17]]

        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 12:18]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [12,13,14,15,16,17,  6,7,8,9,10,11,  0,1,2,3,4,5]]
        
        #Order poles by Re(Pole-position)**2 + Im(Pole-position)**2
        params1 = pole_params[:, 6:12]
        params2 = pole_params[:, 12:18]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,4,5,  12,13,14,15,16,17,  6,7,8,9,10,11]]
        
    return pole_params


def pole_config_organize_abs_dual(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Poles are ordered by Re(Pole-position)**2 + Im(Position)**2 from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=6 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0 or pole_class==1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        
    elif pole_class == 2 or pole_class==3 or pole_class==4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]      *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 6:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,9,10,11,  0,1,2,3,4,5]]
        
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class == 8:  
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]      *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,13]    <  0
        pole_params[indices,13]       *= -1
        pole_params[indices,15]       *= -1
        pole_params[indices,17]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 6:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,9,10,11,  0,1,2,3,4,5,  12,13,14,15,16,17]]

        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 12:18]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [12,13,14,15,16,17,  6,7,8,9,10,11,  0,1,2,3,4,5]]
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 6:12]
        params2 = pole_params[:, 12:18]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,4,5,  12,13,14,15,16,17,  6,7,8,9,10,11]]

    return pole_params


def pole_config_organize_im_dual(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Poles are ordered by Im(Pole-position)**2  from small to large
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
  
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=3 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0 or pole_class==1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        
    elif pole_class == 2 or pole_class==3 or pole_class==4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]      *= -1
        
        #Order poles by Im(Position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 6:12]
        val1    = params1[:,1]**2
        val2    = params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,9,10,11,  0,1,2,3,4,5]]
        
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class == 8:  
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        pole_params[indices,11]      *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,13]    <  0
        pole_params[indices,13]       *= -1
        pole_params[indices,15]       *= -1
        pole_params[indices,17]       *= -1
        
        #Order poles by Im(Position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 6:12]
        val1    = params1[:,1]**2
        val2    = params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [6,7,8,9,10,11,  0,1,2,3,4,5,  12,13,14,15,16,17]]

        #Order poles by Im(Position)**2
        params1 = pole_params[:, 0:6]
        params2 = pole_params[:, 12:18]
        val1    = params1[:,1]**2
        val2    = params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [12,13,14,15,16,17,  6,7,8,9,10,11,  0,1,2,3,4,5]]
        
        #Order poles by Im(Position)**2
        params1 = pole_params[:, 6:12]
        params2 = pole_params[:, 12:18]
        val1    = params1[:,1]**2
        val2    = params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,4,5,  12,13,14,15,16,17,  6,7,8,9,10,11]]

    return pole_params


def remove_zero_imag_parts_dual(pole_class, pole_params):
    """
    Removes columns with zero (or smallest) imaginary parts of poles
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 

    pole_class: int: 0-8
        The pole class
        
    pole_params: numpy.ndarray of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=6 for pole_class=0)
        Pole configurations to be manipulated
    
    returns: numpy.ndarray of shape (m,k'), where m is the number of samples and k' depends on the pole class (e.g. k'=3 for pole_class=0)
        Manipulated pole parameters
    """
    # sort by pos_im
    pole_params = pole_config_organize_im_dual(pole_class, pole_params)
    
    if pole_class == 0:
        pole_params = pole_params[:,[0,2,4]]
    elif pole_class == 2:
        pole_params = pole_params[:,[0,2,4, 6,8,10]]
    elif pole_class == 3:
        # remove imag parts of first pole (smallest pos_im)
        pole_params = pole_params[:,[0,2,4, 6,7,8,9,10,11]]
    elif pole_class == 5:
        pole_params = pole_params[:,[0,2,4, 6,8,10, 12,14,16]]
    elif pole_class == 6:
        # remove imag parts of first pole (smallest pos_im)
        pole_params = pole_params[:,[0,2,4, 6,8,10, 12,13,14,15,16,17]]
    elif pole_class == 7:
        # remove imag parts of first two pole (smallest pos_im)
        pole_params = pole_params[:,[0,2,4, 6,7,8,9,10,11, 12,13,14,15,16,17]]    
    return pole_params


def add_zero_imag_parts_dual(pole_class, pole_params):
    """
    Adds columns with zero imaginary parts to real poles
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 

    pole_class: int: 0-8
        The pole class
        
    pole_params: numpy.ndarray of shape (m,k) or (k,), where m is the number of samples and k depends on the pole class (e.g. k=3 for pole_class=0)
        Pole configurations to be manipulated
    
    returns: numpy.ndarray of shape (m,k'), where m is the number of samples and k' depends on the pole class (e.g. k'=6 for pole_class=0)
        Manipulated pole parameters
    """
    pole_params = np.atleast_2d(pole_params)
    m = np.shape(pole_params)[0]
    zeros = np.zeros((m,1))
    
    if pole_class == 0:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 2:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 3:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3:]
                                ))
    elif pole_class == 5:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros,
                                 pole_params[:,6].reshape(-1,1),
                                 zeros,
                                 pole_params[:,7].reshape(-1,1),
                                 zeros,
                                 pole_params[:,8].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 6:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros,
                                 pole_params[:,6:]
                                ))
    elif pole_class == 7:
        pole_params = np.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3:]
                                ))
    return pole_params


def add_zero_imag_parts_dual_torch(pole_class, pole_params):
    """
    Adds columns with zero imaginary parts to real poles
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 

    pole_class: int: 0-8
        The pole class
        
    pole_params: torch.Tensor of shape (m,k) or (k,), where m is the number of samples and k depends on the pole class (e.g. k=3 for pole_class=0)
        Pole configurations to be manipulated
    
    returns: torch.Tensor of shape (m,k'), where m is the number of samples and k' depends on the pole class (e.g. k'=6 for pole_class=0)
        Manipulated pole parameters
    """
    pole_params = torch.atleast_2d(pole_params)
    m = len(pole_params)
    zeros = torch.zeros((m,1)).to(device=pole_params.device)
    
    if pole_class == 0:
        pole_params = torch.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 2:
        pole_params = torch.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 3:
        pole_params = torch.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3:]
                                ))
    elif pole_class == 5:
        pole_params = torch.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros,
                                 pole_params[:,6].reshape(-1,1),
                                 zeros,
                                 pole_params[:,7].reshape(-1,1),
                                 zeros,
                                 pole_params[:,8].reshape(-1,1),
                                 zeros
                                ))
    elif pole_class == 6:
        pole_params = torch.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3].reshape(-1,1),
                                 zeros,
                                 pole_params[:,4].reshape(-1,1),
                                 zeros,
                                 pole_params[:,5].reshape(-1,1),
                                 zeros,
                                 pole_params[:,6:]
                                ))
    elif pole_class == 7:
        pole_params = torch.hstack((pole_params[:,0].reshape(-1,1),
                                 zeros,
                                 pole_params[:,1].reshape(-1,1),
                                 zeros,
                                 pole_params[:,2].reshape(-1,1),
                                 zeros,
                                 pole_params[:,3:]
                                ))
    return pole_params


def check_inside_bounds_dual(pole_class, pole_params, 
                        re_max, re_min, im_max, im_min, 
                        coeff_re_max, coeff_re_min, 
                        coeff_im_max, coeff_im_min):
    '''
    Checks, whether all values in pole_params are within their given bounds
        
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 

    pole_class: int = 0-8 
        The Class of the Pole Configuration  
        
    pole_params: ndarray of shape (m,k) or (k,), where k depends on the pole_class (e.g k=6 for pole_class=0)
        Parameters specifying the pole configuration
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Boundaries
        
    returns: ndarray of shape (m,) of Bools
    '''
    pole_params = np.atleast_2d(pole_params)
    
    max_params = np.array([re_max,im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max])
    min_params = np.array([re_min,im_min,-coeff_re_max,-coeff_im_max,-coeff_re_max,-coeff_im_max])
    if pole_class in [0,1]:
        min_params = np.tile(min_params, (1))
        max_params = np.tile(max_params, (1))
    elif pole_class in [2,3,4]:
        min_params = np.tile(min_params, (2))
        max_params = np.tile(max_params, (2))
    elif pole_class in [5,6,7,8]:
        min_params = np.tile(min_params, (3))
        max_params = np.tile(max_params, (3))
    else:
        sys.exit("Undefined label.")    
        
    m = np.shape(pole_params)[0]
    max_params = np.tile(max_params, (m,1))
    min_params = np.tile(min_params, (m,1))

    inside_bounds = np.all(min_params <= pole_params,axis=1) * np.all(pole_params <= max_params,axis=1)
    return inside_bounds


def check_inside_bounds_dens_dual(pole_class, pole_params, 
                        re_max, re_min, im_max, im_min, 
                        coeff_re_max, coeff_re_min, 
                        coeff_im_max, coeff_im_min):
    '''
    Checks, whether all values in pole_params are within their given bounds
         
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    "_dens" means, that this function deals with pole configs, where the imaginary parts of real poles have been removed (Without '_dens' in the name these imaginary parts are kept in and are set to zero.) 
       
    pole_class: int = 0-8 
        The Class of the Pole Configuration  
        
    pole_params: ndarray of shape (m,k) or (k,), where k depends on the pole_class (e.g k=3 for pole_class=0)
        Parameters specifying the pole configuration
        
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: numeric
        Boundaries
        
    returns: ndarray of shape (m,) of Bools
    '''
    pole_params = np.atleast_2d(pole_params)

    max_params = np.array([re_max,im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max])
    min_params = np.array([re_min,im_min,-coeff_re_max,-coeff_im_max,-coeff_re_max,-coeff_im_max])
    if pole_class in [0,1]:
        min_params = np.tile(min_params, (1))
        max_params = np.tile(max_params, (1))
    elif pole_class in [2,3,4]:
        min_params = np.tile(min_params, (2))
        max_params = np.tile(max_params, (2))
    elif pole_class in [5,6,7,8]:
        min_params = np.tile(min_params, (3))
        max_params = np.tile(max_params, (3))
    else:
        sys.exit("Undefined label.")    
        
    m = np.shape(pole_params)[0]
    max_params = np.tile(max_params, (m,1))
    min_params = np.tile(min_params, (m,1))

    max_params = remove_zero_imag_parts_dual(pole_class=pole_class, pole_params=max_params)
    min_params = remove_zero_imag_parts_dual(pole_class=pole_class, pole_params=min_params)

    inside_bounds = np.all(min_params <= pole_params,axis=1) * np.all(pole_params <= max_params,axis=1)
    return inside_bounds




