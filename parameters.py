#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Parameters to control the input and behaviour of the code
"""
import pandas as pd
import numpy as np
import os


##############################################################################
##############   General   ###################################################
##############################################################################
# data directory: leave unchanged
data_dir = './data/' 

# sampling grid: the grid on the real axis, that shall be used to create the pole data curves
#                import the grid you want to use or leave as is (then you need to inter-/extrapolate your data to our grid, see file test_data.py)
standard_re = pd.read_csv(os.path.join(data_dir, "integration_gridpoints.csv")).to_numpy().reshape(-1)
# imaginary parts of the grid: leave unchanged
standard_im = np.linspace(0., 0., num=len(standard_re))

# range of real pole positions: adjust to the values which seem suitable for your application
re_max = -0.1
re_min = -10.0

# range of imaginary pole positions (for cc pole pairs only): adjust to the values which seem suitable for your application
im_max = 10.0
im_min = 0.0

# range of absolute (!, the actual values are inside +- this interval) values of pole coefficients (for real and imag part of coeff): 
#    adjust to the values which seem suitable for your application
#    Warning: real poles are represented as cc pole pairs with zero imaginary parts -> 
#             For real poles, the actual coefficient is twice the reported value. (e.g.: 2/x = (1+0*i)/(x+0*i) + (1-0*i)/(x-0*i))
coeff_re_max = 1.0
coeff_re_min = 0.0
coeff_im_max = 1.0
coeff_im_min = 0.0

##############################################################################
##############   Classifier   ################################################
##############################################################################
# Directories: leave unchanged
dir_classifier        = './data_classifier/'
data_dir_classifier   = dir_classifier + 'data/'
log_dir_classifier    = dir_classifier + 'logs/'
models_dir_classifier = dir_classifier + 'models/'

##############################################
############### Data Creation ################
##############################################
# number of data points to be created in the data generation
n_examples_classifier = 100000

# SciPy curve_fit parameters: in the data generation, pole curves are fitted to pole functions and the MSEs and fitted parameters form the ANN input vector
# fitting methods: select SciPy fitting methods  'lm', 'trf' or 'dogbox' (multiple choices of the same are possible)
method_classifier      = ['lm', 'lm', 'trf', 'dogbox', 'dogbox'] 
# use parameter boundaries for the fitting methods: for each fitting methods, it must be specified, whether boundaries are
#       to be used or not. Note: 'lm' cannot use boundaries.
with_bounds_classifier = [False, False, True, True, True] 

##############################################
###############   Training   #################
##############################################
# info about the input to be logged: leave unchanged
temp = [str(i) for i in with_bounds_classifier]
input_name_classifier = [a+'_wb='+b for a,b in zip(method_classifier,temp)]
input_name_classifier = '+'.join(input_name_classifier)

# data split: set to desired values
train_portion_classifier = 0.8
val_portion_classifier   = 0.1
test_portion_classifier  = 0.1

# ANN architecture: select from FC1-FC6 (fully connected feedforward ANN)
architecture_classifier = 'FC2'
# in and out features: leave unchanged
in_features_classifier  = 237*len(method_classifier)
out_features_classifier = 9
# specify the desired number of hidden units per layer
hidden_dim_1_classifier = 64
hidden_dim_2_classifier = 64
hidden_dim_3_classifier = 0
hidden_dim_4_classifier = 0
hidden_dim_5_classifier = 0
hidden_dim_6_classifier = 0
 
# training hyperparameters
# optimizer: select between Adam, AdamW, Adagrad, Adadelta, RMSprop and SGD
optimizer_classifier          = 'Adam'
# specify the batch size and learning rate
batch_size_classifier         = 512
learning_rate_classifier      = 1e-3
# maximal number of epochs: set to lower value, if needed (by default, early stopping is used to stop the runs)
epochs_classifier             = int(1e15)
# validation check intervall = 1/ (number of validation checks per training epoch): set to desired value 
val_check_interval_classifier = 0.1
# early stopping patience: set to desired value
es_patience_classifier        = 1000

# regularization: set weight decay and dropout per hidden layer to the desired values 
#                 see lib/architectures.py for details about dropout implementation
weight_decay_classifier = 0.0    
drop_prob_1_classifier  = 0.0   
drop_prob_2_classifier  = 0.0   
drop_prob_3_classifier  = 0.0   
drop_prob_4_classifier  = 0.0   
drop_prob_5_classifier  = 0.0   
drop_prob_6_classifier  = 0.0   

# do multiple runs?
num_runs_classifier     = 5

##############################################################################
##############   Regressors   ################################################
##############################################################################
# class to be learned
class_regressor = 8

# directories
regressor_subdirs = ['0-1r',
                   '1-1c',
                   '2-2r',
                   '3-1r1c',
                   '4-2c',
                   '5-3r',
                   '6-2r1c',
                   '7-1r2c',
                   '8-3c',]  
dir_regressors       = './data_regressor/'
dir_regressor        = dir_regressors + regressor_subdirs[class_regressor] + '/'
data_dir_regressor   = dir_regressor + 'data/'
log_dir_regressor    = dir_regressor + 'logs/'
models_dir_regressor = dir_regressor + 'models/'

##############################################
############### Data Creation ################
##############################################
# number of data points
n_examples_regressor = 1000000

##############################################
###############   Training   #################
##############################################
# data split
train_portion_regressor = 0.9
val_portion_regressor   = 0.05
test_portion_regressor  = 0.05

# network hyperparameters
# ANN architecture
architecture_regressor = 'FC6'
out_list               = [3,6, 6,9,12, 9,12,15,18]
out_features_regressor = out_list[class_regressor]  # depends on the pole class
in_features_regressor  = len(standard_re)*2
hidden_dim_1_regressor = 256
hidden_dim_2_regressor = hidden_dim_1_regressor
hidden_dim_3_regressor = hidden_dim_2_regressor
hidden_dim_4_regressor = hidden_dim_3_regressor
hidden_dim_5_regressor = hidden_dim_4_regressor
hidden_dim_6_regressor = hidden_dim_5_regressor 

# training hyperparameters
optimizer_regressor          = 'Adam'
batch_size_regressor         = 4096
learning_rate_regressor      = 1e-3   
# maximal number of training epochs
epochs_regressor             = 1000
# maximal number of training steps
max_steps_regressor          = None
val_check_interval_regressor = 1.0
# early stopping patience
es_patience_regressor        = int(1e15)

# loss
parameter_loss_type       = 'mse'
reconstruction_loss_type  = 'mse'
parameter_loss_coeff      = 1.0
reconstruction_loss_coeff = 0.1
loss_name_regressor = (str(parameter_loss_coeff) + '*parameter_loss_' + parameter_loss_type + ' + ' +
                       str(reconstruction_loss_coeff) + '*reconstruction_loss_' + reconstruction_loss_type )

# regularization
weight_decay_regressor = 0.0
drop_prob_1_regressor  = 0.0   
drop_prob_2_regressor  = 0.0   
drop_prob_3_regressor  = 0.0   
drop_prob_4_regressor  = 0.0   
drop_prob_5_regressor  = 0.0   
drop_prob_6_regressor  = 0.0 

# do multiple runs?
num_runs_regressor           = 5





