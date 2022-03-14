# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:34:08 2021

@author: siegfriedkaidisch

Pole objective functions to be used by SciPy's curve_fit

"""
import numpy as np

from lib.pole_objective_functions_single import objective_1r_single, objective_1c_single
from lib.pole_objective_functions_single import objective_2r_single, objective_1r1c_single, objective_2c_single
from lib.pole_objective_functions_single import objective_3r_single, objective_2r1c_single, objective_1r2c_single, objective_3c_single
from lib.pole_objective_functions_single import objective_1r_jac_single, objective_1c_jac_single


def objective_1r_dual(x, a, c1, c2):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c1,2: number
        Real part of the coefficient of the pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1r_single(x=x, a=a, c=c1)
    y2 = objective_1r_single(x=x, a=a, c=c2)
    return np.hstack((y1,y2))

def objective_1c_dual(x, a, b, c1, d1, c2, d2):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1c_single(x, a, b, c1, d1)
    y2 = objective_1c_single(x, a, b, c2, d2)
    return np.hstack((y1,y2))


def objective_2r_dual(x, a, c1, c2, e, g1, g2):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_2r_single(x, a, c1, e, g1)
    y2 = objective_2r_single(x, a, c2, e, g2)
    return np.hstack((y1,y2))


def objective_1r1c_dual(x, a, c1, c2, e, f, g1, h1, g2, h2):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1r1c_single(x, a, c1, e, f, g1, h1)
    y2 = objective_1r1c_single(x, a, c2, e, f, g2, h2)
    return np.hstack((y1,y2))


def objective_2c_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    b: number
        Imaginary part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    d: number
        Imaginary part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_2c_single(x, a, b, c1, d1, e, f, g1, h1)
    y2 = objective_2c_single(x, a, b, c2, d2, e, f, g2, h2)
    return np.hstack((y1,y2))


def objective_3r_dual(x, a, c1, c2, e, g1, g2, i, k1, k2):
    '''
    Objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_3r_single(x, a, c1, e, g1, i, k1)
    y2 = objective_3r_single(x, a, c2, e, g2, i, k2)
    return np.hstack((y1,y2))

def objective_2r1c_dual(x, a, c1, c2, e, g1, g2, i, j, k1, l1, k2, l2):
    '''
    Objective function for SciPy's curve_fit: 2 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_2r1c_single(x, a, c1, e, g1, i, j, k1, l1)
    y2 = objective_2r1c_single(x, a, c2, e, g2, i, j, k2, l2)
    return np.hstack((y1,y2))


def objective_1r2c_dual(x, a, c1, c2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Objective function for SciPy's curve_fit: 1 real pole, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_1r2c_single(x, a, c1, e, f, g1, h1, i, j, k1, l1)
    y2 = objective_1r2c_single(x, a, c2, e, f, g2, h2, i, j, k2, l2)
    return np.hstack((y1,y2))


def objective_3c_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Objective function for SciPy's curve_fit: 0 real poles, 3 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: number or np.ndarray of shape (n,)
        Position, where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    b: number
        Imaginary part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    d: number
        Imaginary part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: number or np.ndarray of shape (2*n,)
        The real part of the function value
    '''
    y1 = objective_3c_single(x, a, b, c1, d1, e, f, g1, h1, i, j, k1, l1)
    y2 = objective_3c_single(x, a, b, c2, d2, e, f, g2, h2, i, j, k2, l2)
    return np.hstack((y1,y2))


def objective_1r_jac_dual(x, a, c1, c2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    zeros = np.zeros((len(x),1))
    
    jac1 = objective_1r_jac_single(x, a, c1)
    jac2 = objective_1r_jac_single(x, a, c2)
    
    jac1 = np.hstack([jac1, zeros])
    jac2 = np.hstack([jac2[:,0].reshape(-1,1), zeros, jac2[:,1].reshape(-1,1)])
    
    jacmat = np.vstack([jac1, jac2])
    return jacmat

def objective_1c_jac_dual(x, a, b, c1, d1, c2, d2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)        
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the pole
        
    b: number
        Imaginary part of the position of the pole
        
    c: number
        Real part of the coefficient of the pole
        
    d: number
        Imaginary part of the coefficient of the pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    zeros = np.zeros((len(x),2))
    
    jac1 = objective_1c_jac_single(x, a, b, c1, d1)
    jac2 = objective_1c_jac_single(x, a, b, c2, d2)
    
    jac1 = np.hstack([jac1, zeros])
    jac2 = np.hstack([jac2[:,0:2], zeros, jac2[:,2:]])
    
    jacmat = np.vstack([jac1, jac2])
    return jacmat

def objective_2r_jac_dual(x, a, c1, c2, e, g1, g2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 2 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)       
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1r_jac_dual(x, e, g1, g2)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_1r1c_jac_dual(x, a, c1, c2, e, f, g1, h1, g2, h2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_2c_jac_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    b: number
        Imaginary part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    d: number
        Imaginary part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1c_jac_dual(x, a, b, c1, d1, c2, d2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jacmat = np.hstack([jac1, jac2])
    return jacmat


def objective_3r_jac_dual(x, a, c1, c2, e, g1, g2, i, k1, k2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 3 real poles, 0 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)          
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1r_jac_dual(x, e, g1, g2)
    jac3   = objective_1r_jac_dual(x, i, k1, k2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_2r1c_jac_dual(x, a, c1, c2, e, g1, g2, i, j, k1, l1, k2, l2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 2 real poles, 1 cc pole pair
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1r_jac_dual(x, e, g1, c2)
    jac3   = objective_1c_jac_dual(x, i, j, k1, l1, k2, l2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_1r2c_jac_dual(x, a, c1, c2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 1 real pole, 2 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)        
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1r_jac_dual(x, a, c1, c2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jac3   = objective_1c_jac_dual(x, i, j, k1, l1, k2, l2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat


def objective_3c_jac_dual(x, a, b, c1, d1, c2, d2, e, f, g1, h1, g2, h2, i, j, k1, l1, k2, l2):
    '''
    Calculates Jacobi matrix for objective function for SciPy's curve_fit: 0 real poles, 3 cc pole pairs
    
    "_dual" means, that this function deals with 2 pole configs with same positions but different coeffs 
    
    x: numpy.ndarray of shape (n,)         
        Position(s), where the function shall be evaluated
        
    a: number
        Real part of the position of the first pole
        
    b: number
        Imaginary part of the position of the first pole
        
    c: number
        Real part of the coefficient of the first pole
        
    d: number
        Imaginary part of the coefficient of the first pole
        
    e: number
        Real part of the position of the second pole
        
    f: number
        Imaginary part of the position of the second pole
        
    g: number
        Real part of the coefficient of the second pole
        
    h: number
        Imaginary part of the coefficient of the second pole
        
    i: number
        Real part of the position of the third pole
        
    j: number
        Imaginary part of the position of the third pole
        
    k: number
        Real part of the coefficient of the third pole
        
    l: number
        Imaginary part of the coefficient of the third pole
        
    returns: numpy.ndarray of shape (2*n,k)
        The Jacobi matrix
    '''
    jac1   = objective_1c_jac_dual(x, a, b, c1, d1, c2, d2)
    jac2   = objective_1c_jac_dual(x, e, f, g1, h1, g2, h2)
    jac3   = objective_1c_jac_dual(x, i, j, k1, l1, k2, l2)
    jacmat = np.hstack([jac1, jac2, jac3])
    return jacmat



















