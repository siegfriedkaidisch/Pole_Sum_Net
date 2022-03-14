# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:03:30 2021

@author: siegfriedkaidisch

Standardization routines

"""
import numpy as np
import sklearn.preprocessing as prepro
import os
from pathlib import Path
import torch


def std_data(data, with_mean, std_path, name_var="variances.npy", name_mean="means.npy"):
    '''
    Apply standardization to data using saved means and variances
    
    data: numpy.ndarray of shape (m,n) or (n,)
        Data to be standardized; can also be multiple (m) samples
        
    with_mean: bool
        Shall the mean of each feature be shifted?
    
    std_path: str
        Path to the folder containing variances and (possibly) means files
    
    name_var, name_mean: str, default="variances.npy", "means.npy"
        Names of the variances and means files
        
    returns: numpy.ndarray of shape (m,n), where m is the number of samples
        The transformed samples are returned
    '''
    data = np.atleast_2d(data)

    variances = np.load(os.path.join(std_path, name_var), allow_pickle=True).astype('float32')
    scaler    = prepro.StandardScaler(with_mean=with_mean)  
    scaler.var_   = variances
    scaler.scale_ = np.sqrt(variances)
    if with_mean:
        means     = np.load(os.path.join(std_path, name_mean), allow_pickle=True).astype('float32')
        scaler.mean_  = means
    data = scaler.transform(data)
    return data


def rm_std_data(data, with_mean, std_path, name_var="variances.npy", name_mean="means.npy"):
    '''
    Remove standardization from data using saved means and variances
    
    data: numpy.ndarray of shape (m,n) or (n,)
        Data to be transformed; can also be multiple (m) samples
        
    with_mean: bool
        Shall the mean of each feature be shifted?
    
    std_path: str
        Path to the folder containing variances and (possibly) means files
    
    name_var, name_mean: str, default="variances.npy", "means.npy"
        Names of the variances and means files
        
    returns: numpy.ndarray of shape (m,n), where m is the number of samples
        The transformed samples are returned
    '''
    variances = np.load(os.path.join(std_path, name_var), allow_pickle=True).astype('float32')
    scaler    = prepro.StandardScaler(with_mean=with_mean)  
    scaler.var_   = variances
    scaler.scale_ = np.sqrt(variances)
    if with_mean:
        means     = np.load(os.path.join(std_path, name_mean), allow_pickle=True).astype('float32')
        scaler.mean_  = means
    data = scaler.inverse_transform(data)
    return data


def std_data_new(data, with_mean, std_path, name_var="variances.npy", name_mean="means.npy"):
    '''
    Create and apply standardization to data
    
    data: numpy.ndarray of shape (m,n) or (n,)
        Data to be standardized; can also be multiple (m) samples
        
    with_mean: bool
        Shall the mean of each feature be shifted?
    
    std_path: str
        Path to the folder, where variances and (possibly) means files shall be stored
    
    name_var, name_mean: str, default="variances.npy", "means.npy"
        Names of the variances and means files
        
    returns: numpy.ndarray of shape (m,n), where m is the number of samples
        The transformed samples are returned
    '''
    data = np.atleast_2d(data)
    
    # Create directory, if it does not already exist
    p = Path(std_path)
    p.mkdir(exist_ok=True, parents=True)
    
    scaler = prepro.StandardScaler(with_mean=with_mean)  
    scaler.fit(data)
    data = scaler.transform(data)
    np.save(os.path.join(std_path, name_var), scaler.var_)   #must be applied when using the trained model
    if with_mean:
        np.save(os.path.join(std_path, name_mean), scaler.mean_)
    return data


def std_data_torch(data, with_mean, std_path, name_var="variances.npy", name_mean="means.npy"):
    '''
    Apply standardization to data using saved means and variances
    
    data: torch.Tensor of shape (m,n) or (n,)
        Data to be standardized; can also be multiple (m) samples
        
    with_mean: bool
        Shall the mean of each feature be shifted?
    
    std_path: str
        Path to the folder containing variances and (possibly) means files
    
    name_var, name_mean: str, default="variances.npy", "means.npy"
        Names of the variances and means files
        
    returns: torch.Tensor of shape (m,n), where m is the number of samples
        The transformed samples are returned
    '''
    data = torch.atleast_2d(data)
    used_device = data.device
    
    variances = torch.from_numpy(np.load(os.path.join(std_path, name_var), allow_pickle=True).astype('float32'))
    scales    = (torch.sqrt(variances)).to(device=used_device)

    if with_mean:
        means = torch.from_numpy(np.load(os.path.join(std_path, name_mean), allow_pickle=True).astype('float32'))
        means = means.to(device=used_device)
        data  = (data - torch.tile(means, (data.shape[0],1))) / torch.tile(scales, (data.shape[0],1))
    else:
        data  = data / torch.tile(scales, (data.shape[0],1)) 
    return data


def rm_std_data_torch(data, with_mean, std_path, name_var="variances.npy", name_mean="means.npy"):
    '''
    Remove standardization from data using saved means and variances
    
    data: torch.Tensor of shape (m,n) or (n,)
        Data to be transformed; can also be multiple (m) samples
        
    with_mean: bool
        Shall the mean of each feature be shifted?
    
    std_path: str
        Path to the folder containing variances and (possibly) means files
    
    name_var, name_mean: str, default="variances.npy", "means.npy"
        Names of the variances and means files
        
    returns: torch.Tensor of shape (m,n), where m is the number of samples
        The transformed samples are returned
    '''
    data = torch.atleast_2d(data)
    used_device = data.device
    
    variances = torch.from_numpy(np.load(os.path.join(std_path, name_var), allow_pickle=True).astype('float32'))
    scales    = (torch.sqrt(variances)).to(device=used_device)

    if with_mean:
        means = torch.from_numpy(np.load(os.path.join(std_path, name_mean), allow_pickle=True).astype('float32'))
        means = means.to(device=used_device)
        data  = data * torch.tile(scales, (data.shape[0],1)) + torch.tile(means, (data.shape[0],1))
    else:
        data  = data * torch.tile(scales, (data.shape[0],1)) 
    return data











