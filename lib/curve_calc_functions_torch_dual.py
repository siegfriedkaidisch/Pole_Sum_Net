# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:03:50 2021

@author: siegfriedkaidisch

Pole functions written in Torch
"""
import torch

from lib.curve_calc_functions_torch_single import complex_conjugate_pole_pair_torch


def pole_curve_calc_torch_dual(pole_class, pole_params, grid_x, device):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: torch.Tensor of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    device: torch.device
        Where (on which device) shall the calculation be done
    
    returns: torch.Tensor of shape (m,2*n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = grid_x.to(device=device)
    pole_params = pole_params.to(device=device)
    
    grid_x = torch.reshape(grid_x, (-1,))  
    pole_params = pole_params.transpose(0,1)
    
    if pole_class in [0,1]:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device)
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[4], pole_params[5], device=device)
    elif pole_class in [2,3,4]:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2],  pole_params[3], device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[4],  pole_params[5], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7], pole_params[8],  pole_params[9], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7], pole_params[10], pole_params[11], device=device)
    elif pole_class in [5,6,7,8]:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1],   pole_params[2],  pole_params[3], device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1],   pole_params[4],  pole_params[5], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7],   pole_params[8],  pole_params[9], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7],   pole_params[10], pole_params[11], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[12], pole_params[13], pole_params[14], pole_params[15], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[12], pole_params[13], pole_params[16], pole_params[17], device=device)                
    return torch.hstack((curve_pred1, curve_pred2))


def pole_curve_calc_torch_dens_dual(pole_class, pole_params, grid_x, device):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    "_dens" means, that this function deals with pole configs, where the imaginary parts of real poles have been removed (Without '_dens' in the name these imaginary parts are kept in and are set to zero.) 
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: torch.Tensor of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    device: torch.device
        Where (on which device) shall the calculation be done
    
    returns: torch.Tensor of shape (m,2*n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = grid_x.to(device=device)
    pole_params = pole_params.to(device=device)
    
    grid_x = torch.reshape(grid_x, (-1,))  
    pole_params = pole_params.transpose(0,1)
    
    if pole_class == 0:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device)
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[2], pole_params[0]*0, device=device)
    elif pole_class == 1:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[4], pole_params[5], device=device)
    elif pole_class == 2:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[2], pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[0]*0, pole_params[4], pole_params[0]*0, device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[0]*0, pole_params[5], pole_params[0]*0, device=device)
    elif pole_class == 3:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[2], pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[4],   pole_params[5], pole_params[6], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[4],   pole_params[7], pole_params[8], device=device)
    elif pole_class == 4:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2],  pole_params[3], device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[4],  pole_params[5], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7], pole_params[8],  pole_params[9], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7], pole_params[10], pole_params[11], device=device)
    elif pole_class == 5:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[2], pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[0]*0, pole_params[4], pole_params[0]*0, device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[0]*0, pole_params[5], pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[0]*0, pole_params[7], pole_params[0]*0, device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[0]*0, pole_params[8], pole_params[0]*0, device=device)
    elif pole_class == 6:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[2], pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[0]*0, pole_params[4], pole_params[0]*0, device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[0]*0, pole_params[5], pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7],   pole_params[8], pole_params[9], device=device)
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7],   pole_params[10],pole_params[11], device=device)         
    elif pole_class == 7:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1],  pole_params[0]*0, device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[2],  pole_params[0]*0, device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[4],   pole_params[5],  pole_params[6], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[3], pole_params[4],   pole_params[7],  pole_params[8], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[9], pole_params[10],  pole_params[11], pole_params[12], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[9], pole_params[10],  pole_params[13], pole_params[14], device=device)          
    elif pole_class == 8:
        curve_pred1  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0],  pole_params[1],  pole_params[2],   pole_params[3], device=device) 
        curve_pred2  = complex_conjugate_pole_pair_torch(grid_x, pole_params[0],  pole_params[1],  pole_params[4],   pole_params[5], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6],  pole_params[7],  pole_params[8],   pole_params[9], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[6],  pole_params[7],  pole_params[10],  pole_params[11], device=device)
        curve_pred1 += complex_conjugate_pole_pair_torch(grid_x, pole_params[12], pole_params[13], pole_params[14],  pole_params[15], device=device) 
        curve_pred2 += complex_conjugate_pole_pair_torch(grid_x, pole_params[12], pole_params[13], pole_params[16],  pole_params[17], device=device)
    return torch.hstack((curve_pred1, curve_pred2))







