# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:03:50 2021

@author: siegfriedkaidisch

Pole functions written in Torch
"""
import torch


def single_complex_pole_torch(x_re, x_im, part_re, part_im, coeff_re, coeff_im, device, power=1):
    """
    Computes the function values on a list of complex coordinates of a 
    single singularity of power power, with a multiplicative complex coefficient 
    and a position in the complex plane given by part_re and part_im
    
    x_re, x_im: torch.Tensor of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re, part_im: torch.Tensor of shape (m,) or (1,m)
        Pole positions
        
    coeff_re, coeff_im: torch.Tensor of shape (m,) or (1,m)
        Pole coefficients
        
    device: torch.device
        Where (on which device) shall the calculation be done
        
    power: number, default=1
        The power of the poles
        
    returns: two torch.Tensors of shape (m,n)
        The real and imaginary parts of the function values
    """
    x_re     = x_re.to(device=device)
    x_im     = x_im.to(device=device)
    part_re  = part_re.to(device=device)
    part_im  = part_im.to(device=device)
    coeff_re = coeff_re.to(device=device)
    coeff_im = coeff_im.to(device=device)
    
    x_re = torch.reshape(x_re, (-1,))
    x_im = torch.reshape(x_im, (-1,))
    part_re = torch.reshape(part_re, (-1,))
    part_im = torch.reshape(part_im, (-1,))
    coeff_re = torch.reshape(coeff_re, (-1,))
    coeff_im = torch.reshape(coeff_im, (-1,))
 
    num_params = len(part_re)
    num_points = len(x_re)
    
    diff = ((torch.tile(x_re, (num_params,1)) - torch.tile(torch.reshape(part_re,(num_params,1)),(1, num_points))) + 
        1j * (torch.tile(x_im, (num_params,1)) - torch.tile(torch.reshape(part_im,(num_params,1)),(1, num_points))) )
    
    result = (torch.tile(torch.reshape(coeff_re,(num_params,1)),(1, num_points)) + 1j*torch.tile(torch.reshape(coeff_im,(num_params,1)),(1, num_points)))/diff**power
    result_re = torch.real(result)
    result_im = torch.imag(result)
    
    return result_re, result_im


def complex_conjugate_pole_pair_torch(x_re, part_re, part_im, coeff_re, coeff_im, device, power=1):
    """
    Computes the function values on a list of coordinates on the real axis of a
    singularity of power power plus the values from the conjugate pole, 
    with a multiplicative, complex coefficient and a position on the real axis 
    given by part_re and on the imaginary axis given by part_im 
    
    x_re: torch.Tensor of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re, part_im: torch.Tensor of shape (m,) or (1,m)
        Pole positions
        
    coeff_re, coeff_im: torch.Tensor of shape (m,) or (1,m)
        Pole coefficients
        
    device: torch.device
        Where (on which device) shall the calculation be done
        
    power: number, default=1
        The power of the poles
        
    returns: torch.Tensor of shape (m,n)
        The real part of the function values
    """
    x_re     = x_re.to(device=device)
    part_re  = part_re.to(device=device)
    part_im  = part_im.to(device=device)
    coeff_re = coeff_re.to(device=device)
    coeff_im = coeff_im.to(device=device)
    
    x_re = torch.reshape(x_re, (-1,))
    part_re = torch.reshape(part_re, (-1,))
    part_im = torch.reshape(part_im, (-1,))
    coeff_re = torch.reshape(coeff_re, (-1,))
    coeff_im = torch.reshape(coeff_im, (-1,))
    
    len_set = len(x_re)
    x_im = torch.linspace(0., 0., steps=len_set)
    x_im = x_im.to(device=device)
    
    result_re_plus, _ = single_complex_pole_torch(x_re, x_im, part_re, part_im, coeff_re, coeff_im, device=device, power=power)
    result_re_minus, _ = single_complex_pole_torch(x_re, x_im, part_re=part_re, part_im=-part_im, coeff_re=coeff_re, coeff_im=-coeff_im, device=device, power=power)

    return result_re_plus + result_re_minus


def pole_curve_calc_torch_single(pole_class, pole_params, grid_x, device):
    '''
    Calculate the real part of given pole configurations on a given grid
    
    "_single" means, that this function deals with only 1 pole config
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: torch.Tensor of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    device: torch.device
        Where (on which device) shall the calculation be done
    
    returns: torch.Tensor of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = grid_x.to(device=device)
    pole_params = pole_params.to(device=device)
    
    grid_x = torch.reshape(grid_x, (-1,))  
    pole_params = pole_params.transpose(0,1)
    
    if pole_class in [0,1]:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device)
    elif pole_class in [2,3,4]:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7], device=device)
    elif pole_class in [5,6,7,8]:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[8], pole_params[9], pole_params[10], pole_params[11], device=device)
    return curve_pred


def pole_curve_calc_torch_dens_single(pole_class, pole_params, grid_x, device):
    '''
    Calculate the real part of given pole configurations on a given grid

    "_single" means, that this function deals with only 1 pole config

    "_dens" means, that this function deals with pole configs, where the imaginary parts of real poles have been removed (Without '_dens' in the name these imaginary parts are kept in and are set to zero.) 
    
    pole_class: int = 0-8 
        The class of the pole configuration
    
    pole_params: torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole_class (e.g k=2 for pole_class=0)
        Parameters specifying the pole configuration
    
    grid_x: torch.Tensor of shape (n,) or (1,n)
        Gridpoints, where the function/pole configuration shall be evaluated
        
    device: torch.device
        Where (on which device) shall the calculation be done
    
    returns: torch.Tensor of shape (m,n)
        Function values, i.e. the 'y-values'
    '''
    grid_x = grid_x.to(device=device)
    pole_params = pole_params.to(device=device)
    
    grid_x = torch.reshape(grid_x, (-1,))  
    pole_params = pole_params.transpose(0,1)
    
    if pole_class == 0:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device)
    elif pole_class == 1:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device)
    elif pole_class == 2:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[2], pole_params[0]*0, pole_params[3], pole_params[0]*0, device=device)
    elif pole_class == 3:
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[2], pole_params[3], pole_params[4], pole_params[5], device=device)
    elif pole_class == 4:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7], device=device)
    elif pole_class == 5:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[2], pole_params[0]*0, pole_params[3], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[4], pole_params[0]*0, pole_params[5], pole_params[0]*0, device=device)
    elif pole_class == 6:
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[2], pole_params[0]*0, pole_params[3], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7], device=device)
    elif pole_class == 7:
        curve_pred = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[0]*0, pole_params[1], pole_params[0]*0, device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[2], pole_params[3], pole_params[4], pole_params[5], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[6], pole_params[7], pole_params[8], pole_params[9], device=device)
    elif pole_class == 8:
        curve_pred   = complex_conjugate_pole_pair_torch(grid_x, pole_params[0], pole_params[1], pole_params[2], pole_params[3], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[4], pole_params[5], pole_params[6], pole_params[7], device=device) + complex_conjugate_pole_pair_torch(grid_x, pole_params[8], pole_params[9], pole_params[10], pole_params[11], device=device)
    return curve_pred







