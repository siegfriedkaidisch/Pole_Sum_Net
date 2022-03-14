#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Regressor based on pytorch basic template
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import matplotlib.pyplot as plt

from lib.architectures import FC1, FC2, FC3, FC4, FC5, FC6
from lib.standardization_functions import rm_std_data_torch, std_data_torch

from lib.curve_calc_functions_torch_dual import pole_curve_calc_torch_dens_dual as pole_curve_calc
from lib.pole_config_organize_dual import pole_config_organize_abs_dens_dual as pole_config_organize

from lib.pole_config_organize_dual import add_zero_imag_parts_dual_torch
from lib.pole_config_organize_dual import pole_config_organize_abs_dual
from lib.curve_calc_functions_torch_dual import pole_curve_calc_torch_dual

##############################################################################
###################   Data-related Classes   #################################
##############################################################################
class PoleDataSet_Regressor(Dataset):
    """
    DataSet of the Pole Regressor

    data_dir: str
        Directory containing data files
    
    num_use_data: int
        How many of the available datapoints shall be used?
    """
    def __init__(self, data_dir, test_portion):
        data_X = np.load(os.path.join(data_dir, 'various_poles_data_regressor_x.npy'), allow_pickle=True).astype('float32')
        data_Y = np.load(os.path.join(data_dir, 'various_poles_data_regressor_y.npy'), allow_pickle=True).astype('float32')
       
        print("Checking shape of loaded data: X: ", np.shape(data_X))
        print("Checking shape of loaded data: y: ", np.shape(data_Y))
        self.real_len = len(data_Y)
        
        # split off testing data
        #seed_afterward = np.random.randint(low=0, high=1e3)
        #np.random.seed(1234)
        num_data = len(data_Y)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        
        test_indices      = indices[0:int(num_data*test_portion)]
        train_val_indices = indices[int(num_data*test_portion):]
        #np.random.seed(seed_afterward)
        self.train_val_data_X = data_X[train_val_indices]
        self.train_val_data_Y = data_Y[train_val_indices]
        
        self.test_data_X = data_X[test_indices]
        self.test_data_Y = data_Y[test_indices]
  
    def __len__(self):
        num_of_data_points = len(self.train_val_data_Y)
        return num_of_data_points

    def __getitem__(self, idx):
        return self.train_val_data_X[idx], self.train_val_data_Y[idx]
     
        
class PoleDataModule_Regressor(pl.LightningDataModule):
    """
    DataModule of the Pole Regressor
    
    data_dir: str
        Directory containing data files
    
    batch_size: int
    
    train_portion, validation_portion, test_portion: float
    
    num_use_data: int
        How many of the available datapoints shall be used?
    """
    def __init__(self, data_dir: str, batch_size: int, train_portion: float, validation_portion: float, test_portion: float):
        super().__init__()
        self.data_dir           = data_dir
        self.batch_size         = batch_size
        self.train_portion      = train_portion
        self.validation_portion = validation_portion
        self.test_portion       = test_portion

    def setup(self, stage):
        self.all_data = PoleDataSet_Regressor(data_dir=self.data_dir, test_portion=self.test_portion)
        
        num_data = self.all_data.real_len
        print("Length of all_data: ", num_data)
        
        self.test_number       = int(self.test_portion*num_data)
        self.val_number = int(self.validation_portion*num_data)
        self.train_number       = int(num_data - self.test_number - self.val_number)      
        print("Data splits: ", self.train_number, self.val_number, self.test_number, num_data)
        
        train_part, val_part  = random_split(self.all_data, [self.train_number, self.val_number]
                                                       )#, generator=torch.Generator().manual_seed(1234))

        self.train_dataset = train_part
        self.val_dataset = val_part

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,   batch_size=self.batch_size)

    
##############################################################################
###########################   Regressor    ###################################
##############################################################################

class pole_reconstruction_loss(torch.nn.Module):
    '''
    Reconstruction Loss for Pole Regressor
    
    y_hat: torch.Tensor of shape (batch_size, out_features_regressor)
        Parameter predictions 
        
    x: torch.Tensor of shape (batch_size, in_features_regressor)
        Actual pole curves 
        
    pole_class: int: 0-8
        The pole class
        
    std_path: str
        Path to the folder, where variances and means files shall be stored
        
    grid_x: np.ndarray or torch.Tensor of shape (in_features_regressor,) or (1,in_features_regressor)
        The integration grid
        
    loss_type: str: 'mse' or 'l1'
        
    return: scalar torch.Tensor
        The MSE Reconstruction Loss
    '''
    def __init__(self, pole_class, std_path, grid_x, loss_type):
        super(pole_reconstruction_loss,self).__init__()
        self.pole_class = pole_class
        self.std_path   = std_path
        self.grid_x     = grid_x
        self.loss_type  = loss_type
        
    def forward(self,y_hat, x):
        # Remove std from x and y_hat
        x     = rm_std_data_torch(data=x, with_mean=False, 
                                        std_path=self.std_path, name_var="variances.npy")
        y_hat = rm_std_data_torch(data=y_hat, with_mean=True, 
                                        std_path=self.std_path, name_var='variances_params.npy', name_mean='means_params.npy')

        # Calculate Pole curves from y_hat
        x_pred = pole_curve_calc(pole_class=self.pole_class, pole_params=y_hat, grid_x=self.grid_x, device=y_hat.device)
        
        # Calculate MSE
        if self.loss_type == 'mse':
            loss   = F.mse_loss(x_pred, x)
        elif self.loss_type == 'l1':
            loss   = F.l1_loss(x_pred, x)
            
        #plt.plot(x[0,:].cpu().detach().numpy())
        #plt.plot(x_pred[0,:].cpu().detach().numpy())
        #plt.show()
            
        return loss


class Pole_Regressor(LightningModule):
    """
    Basic lightning model to use a vector of inputs in order to predict
    the parameters of a complex structure in the vector
    
    pole_class: int
        The class that shall be trained
        
    std_path: str
        Folder containing the standardization info of the data (needed for special loss)
    
    re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min: floats
        The parameter boundaries
    
    grid_x: np.ndarray or torch.Tensor of shape (in_features_regressor,) or (1,in_features_regressor)
        The integration grid
    
    weight_decay, learning_rate: floats
    
    architecture, optimizer: strings
    
    additional kwargs are handed to the architecture class
    """
    def __init__(self, 
                 # The pole class
                 pole_class: int,
                 
                 # Path to the folder, where variances and means files shall be stored
                 std_path: str,
                 
                 # The Integration grid
                 grid_x,
                 
                 # Parameter boundaries
                 re_max, re_min, im_max, im_min, 
                 coeff_re_max, coeff_re_min, 
                 coeff_im_max, coeff_im_min,
                 
                 # Specify the loss
                 parameter_loss_type, reconstruction_loss_type,
                 parameter_loss_coeff, reconstruction_loss_coeff,
                 
                 # Regularization
                 weight_decay:  float = 0.0,
                 
                 # Training-related hyperparameters
                 learning_rate: float = 1e-3, 
                 
                 # ANN Architecture
                 architecture: str = 'FC6', 
                 
                 # Optimizer
                 optimizer: str = 'Adam',
                 
                 #Additional arguments needed for initialization of the ANN Architecture
                 **kwargs
                 ):
        
        super().__init__()
        
        # prepare grid_x
        if type(grid_x) == np.ndarray:
            grid_x = torch.from_numpy(grid_x)
        grid_x = grid_x.to(device=self.device).float()
        
        self.save_hyperparameters()
        self.model = globals()[architecture](**kwargs)
        
        self.Reconstruction_loss_mse = pole_reconstruction_loss(pole_class=pole_class, std_path=std_path, grid_x=self.hparams.grid_x, 
                                                                loss_type='mse')
        self.Reconstruction_loss_l1  = pole_reconstruction_loss(pole_class=pole_class, std_path=std_path, grid_x=self.hparams.grid_x, 
                                                                loss_type='l1')
        self.Parameter_loss_mse      = nn.MSELoss()
        self.Parameter_loss_l1       = nn.L1Loss()
        
        self.boundary_setup_finished = False
        
    def losses(self, x, y, y_hat):
        parameter_loss_type            = self.hparams.parameter_loss_type
        reconstruction_loss_type       = self.hparams.reconstruction_loss_type
        parameter_loss_coeff           = self.hparams.parameter_loss_coeff
        reconstruction_loss_coeff      = self.hparams.reconstruction_loss_coeff
        
        if parameter_loss_type        == 'mse':
            parameter_loss             = self.Parameter_loss_mse(y_hat, y)
        elif parameter_loss_type      == 'l1':
            parameter_loss             = self.Parameter_loss_l1(y_hat, y)
        if reconstruction_loss_type   == 'mse':
            reconstruction_loss        = self.Reconstruction_loss_mse(y_hat, x)  
        elif reconstruction_loss_type == 'l1':
            reconstruction_loss        = self.Reconstruction_loss_l1(y_hat, x)  

        # loss that is used to train the network:
        loss                           = (parameter_loss_coeff*parameter_loss + 
                                          reconstruction_loss_coeff*reconstruction_loss)
        return [parameter_loss, reconstruction_loss, loss]
 
    def forward(self, x):
        x = self.model(x)
        x = self.apply_boundaries(x)
        x = self.pole_config_organize(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self(x)             
        _, _, loss = self.losses(x=x,y=y,y_hat=y_hat)
        self.log('train_loss', loss, on_step=True, on_epoch=False) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)     
        _, _, loss = self.losses(x=x,y=y,y_hat=y_hat)
        self.log('val_loss', loss, on_step=False, on_epoch=True) 
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}
    
    def apply_boundaries(self, x):
        if self.boundary_setup_finished == False:
            # Get lower and upper bounds: 
            re_max = self.hparams.re_max
            re_min = float('-inf')
            im_max = float('inf')
            im_min = float('-inf')
            coeff_re_max = float('inf')
            coeff_im_max = float('inf')
            max_params = torch.Tensor([re_max,im_max, coeff_re_max, coeff_im_max, coeff_re_max, coeff_im_max])
            min_params = torch.Tensor([re_min,im_min,-coeff_re_max,-coeff_im_max,-coeff_re_max,-coeff_im_max])
            if self.hparams.pole_class in [0,1]:
                min_params = torch.tile(min_params, (1,1))
                max_params = torch.tile(max_params, (1,1))
            elif self.hparams.pole_class in [2,3,4]:
                min_params = torch.tile(min_params, (1,2))
                max_params = torch.tile(max_params, (1,2))
            elif self.hparams.pole_class in [5,6,7,8]:
                min_params = torch.tile(min_params, (1,3))
                max_params = torch.tile(max_params, (1,3))
            else:
                sys.exit("Undefined label.")    
                
            # remove imag parts of real poles    
            if   self.hparams.pole_class == 0:
                max_params = max_params[:,[0,2,4]]
                min_params = min_params[:,[0,2,4]]
            elif self.hparams.pole_class == 2:
                max_params = max_params[:,[0,2,4, 6,8,10]]
                min_params = min_params[:,[0,2,4, 6,8,10]]
            elif self.hparams.pole_class == 3:
                max_params = max_params[:,[0,2,4, 6,7,8,9,10,11]]
                min_params = min_params[:,[0,2,4, 6,7,8,9,10,11]]
            elif self.hparams.pole_class == 5:
                max_params = max_params[:,[0,2,4, 6,8,10, 12,14,16]]
                min_params = min_params[:,[0,2,4, 6,8,10, 12,14,16]]
            elif self.hparams.pole_class == 6:
                max_params = max_params[:,[0,2,4, 6,8,10, 12,13,14,15,16,17]]
                min_params = min_params[:,[0,2,4, 6,8,10, 12,13,14,15,16,17]]
            elif self.hparams.pole_class == 7:
                max_params = max_params[:,[0,2,4, 6,7,8,9,10,11, 12,13,14,15,16,17]]
                min_params = min_params[:,[0,2,4, 6,7,8,9,10,11, 12,13,14,15,16,17]]
                
            #Apply the standardization to the boundaries 
            min_params = std_data_torch(data=min_params, with_mean=True, 
                              std_path=self.hparams.std_path, name_var='variances_params.npy', name_mean='means_params.npy')
            max_params = std_data_torch(data=max_params, with_mean=True, 
                              std_path=self.hparams.std_path, name_var='variances_params.npy', name_mean='means_params.npy') 
            self.min_params = min_params.to(device=x.device)
            self.max_params = max_params.to(device=x.device)
            self.boundary_setup_finished = True  
            
        min_params = torch.tile(self.min_params, (len(x),1))
        max_params = torch.tile(self.max_params, (len(x),1))
            
        # Apply boundaries
        x = torch.maximum(x, min_params)
        x = torch.minimum(x, max_params)
        return x
    
    def pole_config_organize(self, x):
        # Remove std
        x = rm_std_data_torch(data=x, with_mean=True, 
                              std_path=self.hparams.std_path, name_var='variances_params.npy', name_mean='means_params.npy')
        # Sort Poles
        x = pole_config_organize(pole_class=self.hparams.pole_class, pole_params=x)
        # Apply std
        x = std_data_torch(data=x, with_mean=True, 
                              std_path=self.hparams.std_path, name_var='variances_params.npy', name_mean='means_params.npy')
        return x




#