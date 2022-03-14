# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:08:24 2021

@author: siegfriedkaidisch

Pole functions for data generation

"""
import numpy as np


def single_complex_pole(x_re, x_im, part_re, part_im, coeff_re, coeff_im, power=1):
    """
    Computes the function values on a list of complex coordinates of a 
    single singularity of power power, with a multiplicative complex coefficient 
    and a position in the complex plane given by part_re and part_im
    
    x_re, x_im: numpy.ndarray of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re, part_im: numpy.ndarray of shape (m,) or (1,m)
        Pole positions
        
    coeff_re, coeff_im: numpy.ndarray of shape (m,) or (1,m)
        Pole coefficients
        
    returns: two numpy.ndarrays of shape (m,n)
        The real and imaginary parts of the function values
    """
    x_re = np.reshape(x_re, -1)
    x_im = np.reshape(x_im, -1)
    part_re = np.reshape(part_re, -1)
    part_im = np.reshape(part_im, -1)
    coeff_re = np.reshape(coeff_re, -1)
    coeff_im = np.reshape(coeff_im, -1)
 
    num_params = len(part_re)
    num_points = len(x_re)
    
    diff = ((np.tile(x_re, (num_params,1)) - np.tile(np.reshape(part_re,(num_params,1)),(1, num_points))) + 
        1j * (np.tile(x_im, (num_params,1)) - np.tile(np.reshape(part_im,(num_params,1)),(1, num_points))) )
    
    result = (np.tile(np.reshape(coeff_re,(num_params,1)),(1, num_points)) + 1j*np.tile(np.reshape(coeff_im,(num_params,1)),(1, num_points)))/diff**power
    result_re = np.real(result)
    result_im = np.imag(result)
    
    return result_re, result_im


def single_real_pole(x_re, part_re, coeff_re, power=1):
    """
    Computes the function values on a list of coordinates on the real axis of a
    single singularity of power power, with a multiplicative, real coefficient
    and a position on the real axis given by part_re 
    
    x_re: numpy.ndarray of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re: numpy.ndarray of shape (m,) or (1,m)
        Pole positions
        
    coeff_re: numpy.ndarray of shape (m,) or (1,m)
        Pole coefficients
        
    power: number
        The power of the poles
        
    returns: numpy.ndarray of shape (m,n)
        The real part of the function values
    """
    x_re = np.reshape(x_re, -1)
    part_re = np.reshape(part_re, -1)
    coeff_re = np.reshape(coeff_re, -1)
    
    len_set = len(x_re)
    x_im = np.linspace(0., 0., num=len_set)
    
    result_re, _ = single_complex_pole(x_re, x_im, part_re=part_re, part_im=np.zeros(np.shape(part_re)), coeff_re=coeff_re, coeff_im=np.zeros(np.shape(coeff_re)), power=power)
    
    return result_re


def complex_conjugate_pole_pair(x_re, part_re, part_im, coeff_re, coeff_im, power=1):
    """
    Computes the function values on a list of coordinates on the real axis of a
    singularity of power power plus the values from the conjugate pole, 
    with a multiplicative, complex coefficient and a position on the real axis 
    given by part_re and on the imaginary axis given by part_im 
    
    x_re: numpy.ndarray of shape (n,) or (1,n)
        Positions, where the function shall be evaluated
        
    part_re, part_im: numpy.ndarray of shape (m,) or (1,m)
        Pole positions
        
    coeff_re, coeff_im: numpy.ndarray of shape (m,) or (1,m)
        Pole coefficients
        
    power: number
        The power of the poles
        
    returns: numpy.ndarray of shape (m,n)
        The real part of the function values
    """
    x_re = np.reshape(x_re, -1)
    part_re = np.reshape(part_re, -1)
    part_im = np.reshape(part_im, -1)
    coeff_re = np.reshape(coeff_re, -1)
    coeff_im = np.reshape(coeff_im, -1)
    
    len_set = len(x_re)
    x_im = np.linspace(0., 0., num=len_set)
    
    result_re_plus, _ = single_complex_pole(x_re, x_im, part_re, part_im, coeff_re, coeff_im, power=power)
    result_re_minus, _ = single_complex_pole(x_re, x_im, part_re=part_re, part_im=-part_im, coeff_re=coeff_re, coeff_im=-coeff_im, power=power)

    return result_re_plus + result_re_minus

