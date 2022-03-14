#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Regressor based on pytorch basic template: Training file
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import time
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
import argparse
import torch

from lib.pole_regressor import Pole_Regressor, PoleDataModule_Regressor
from lib.standardization_functions import rm_std_data
from lib.pole_config_organize_dual import add_zero_imag_parts_dual, pole_config_organize_abs_dual
from lib.diverse_functions import mse, mae

from parameters import standard_re
from parameters import data_dir_regressor, models_dir_regressor, log_dir_regressor
from parameters import train_portion_regressor, val_portion_regressor, test_portion_regressor
from parameters import architecture_regressor, in_features_regressor, out_features_regressor
from parameters import hidden_dim_1_regressor, hidden_dim_2_regressor, hidden_dim_3_regressor, hidden_dim_4_regressor, hidden_dim_5_regressor, hidden_dim_6_regressor
from parameters import epochs_regressor, learning_rate_regressor, weight_decay_regressor
from parameters import batch_size_regressor
from parameters import drop_prob_1_regressor, drop_prob_2_regressor, drop_prob_3_regressor
from parameters import drop_prob_4_regressor, drop_prob_5_regressor, drop_prob_6_regressor
from parameters import optimizer_regressor
from parameters import val_check_interval_regressor, es_patience_regressor
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import n_examples_regressor
from parameters import num_runs_regressor
from parameters import class_regressor
from parameters import parameter_loss_type, reconstruction_loss_type
from parameters import parameter_loss_coeff, reconstruction_loss_coeff
from parameters import loss_name_regressor
from parameters import max_steps_regressor


##############################################################################
##########################   Execution   #####################################
##############################################################################
if __name__ == '__main__':
    ##########   Parse hyperparameters, if wanted   ##########################
    #'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_regressor', type=float, default=batch_size_regressor)
    parser.add_argument('--learning_rate_regressor', type=int, default=learning_rate_regressor)  
    args = parser.parse_args()
    batch_size_regressor = args.batch_size_regressor
    learning_rate_regressor = args.learning_rate_regressor
    #'''
    ##########################################################################
    time1 = time.time()
    seed_everything(seed=1234)  #standard: 1234
    seeds = np.random.randint(0,1e6,size=[num_runs_regressor]) 
    
    net_hyperparameters = dict(
                pole_class = class_regressor,
                
                std_path = data_dir_regressor,
                
                grid_x=standard_re,
                
                architecture = architecture_regressor,
                in_features  = in_features_regressor,
                out_features = out_features_regressor, 
                hidden_dim_1 = hidden_dim_1_regressor,
                hidden_dim_2 = hidden_dim_2_regressor,
                hidden_dim_3 = hidden_dim_3_regressor,
                hidden_dim_4 = hidden_dim_4_regressor,
                hidden_dim_5 = hidden_dim_5_regressor,
                hidden_dim_6 = hidden_dim_6_regressor,
                
                weight_decay  = weight_decay_regressor,  
                drop_prob_1   = drop_prob_1_regressor,
                drop_prob_2   = drop_prob_2_regressor,
                drop_prob_3   = drop_prob_3_regressor,
                drop_prob_4   = drop_prob_4_regressor,
                drop_prob_5   = drop_prob_5_regressor,
                drop_prob_6   = drop_prob_6_regressor,
                
                re_max = re_max, 
                re_min = re_min, 
                im_max = im_max, 
                im_min = im_min, 
                coeff_re_max = coeff_re_max, 
                coeff_re_min = coeff_re_min, 
                coeff_im_max = coeff_im_max, 
                coeff_im_min = coeff_im_min,
                
                learning_rate = learning_rate_regressor,
                optimizer     = optimizer_regressor,
                
                parameter_loss_type       = parameter_loss_type,
                reconstruction_loss_type  = reconstruction_loss_type,
                parameter_loss_coeff      = parameter_loss_coeff,
                reconstruction_loss_coeff = reconstruction_loss_coeff
		)
    
    other_hyperparameters = dict(
                n_examples_regressor    = n_examples_regressor,
                train_portion_regressor = train_portion_regressor,
                val_portion_regressor   = val_portion_regressor,
                test_portion_regressor  = test_portion_regressor, 
                batch_size              = batch_size_regressor,
                
                val_check_interval      = val_check_interval_regressor,
                es_patience             = es_patience_regressor,
                
                loss_name_regressor     = loss_name_regressor,
                max_steps_regressor     = max_steps_regressor
		)
    
    hyperparameters = {**net_hyperparameters, **other_hyperparameters}

    test_params_rmse_list = []
    test_params_mae_list  = []
    test_params_rmses_list = []
    test_params_maes_list  = []
    for i in range(num_runs_regressor): # average test_acc over multiple runs
        seed_everything(seed=seeds[i])
        name   = 'classifier_run_' + str(time.time())
        
        #wandb.init(config=hyperparameters,
        #           entity="ml-tpp", project="pole_classifier",
        #           group="",
        #           notes="",
        #           tags = ["Regressor"])

        #logger = WandbLogger(save_dir=log_dir_regressor, name=name) 
        logger = TensorBoardLogger(save_dir=log_dir_regressor, name=name)    

        model = Pole_Regressor(
                    **net_hyperparameters
                    )
            
        datamodule = PoleDataModule_Regressor(data_dir=data_dir_regressor, batch_size=batch_size_regressor, 
                                train_portion=train_portion_regressor, validation_portion=val_portion_regressor, test_portion=test_portion_regressor)
        
        checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
            dirpath = models_dir_regressor,
            #filename = 'name_' + str(wandb.run.name) + '_id_' + str(wandb.run.id) + '_' + str(i),
            filename = name,
            monitor="val_loss",
            save_top_k=1, 
            mode='min',
            save_last= False
        )
        
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=es_patience_regressor, mode="min")
        
        trainer = pl.Trainer(
            logger=logger,
            val_check_interval=val_check_interval_regressor,
            callbacks=[checkpoint_callback1, early_stop_callback],
            max_epochs=epochs_regressor,
            max_steps = max_steps_regressor,
            gpus=1
        )
    
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)
     
        #######################################################################
        #######################################################################  
        # Manually do test step     
        test_data_X = torch.from_numpy(datamodule.all_data.test_data_X)
        test_data_Y = datamodule.all_data.test_data_Y
        model = Pole_Regressor.load_from_checkpoint(checkpoint_path=models_dir_regressor+name+'.ckpt')
        model.eval()
        model.to(device='cpu')
        
        preds = model(test_data_X).cpu().detach().numpy()
        
        # Log Test errors (params RMSEs) without std (remove std)
        # Remove std from y and y_hat
        y     = rm_std_data(data=test_data_Y, with_mean=True, 
                                        std_path=data_dir_regressor, name_var='variances_params.npy', name_mean='means_params.npy')
        y_hat = rm_std_data(data=preds, with_mean=True, 
                                        std_path=data_dir_regressor, name_var='variances_params.npy', name_mean='means_params.npy')
        
        #Add zero imaginary parts for real poles
        y     = add_zero_imag_parts_dual(pole_class=class_regressor, pole_params=y)
        y_hat = add_zero_imag_parts_dual(pole_class=class_regressor, pole_params=y_hat)
        # sort poles by abs of position
        y     = pole_config_organize_abs_dual(pole_class=class_regressor, pole_params=y)
        y_hat = pole_config_organize_abs_dual(pole_class=class_regressor, pole_params=y_hat)  

        # Parameters RMSE 
        params_rmse = np.sqrt(mse(y_hat, y)) 
        trainer.logger.log_metrics({'test_params_rmse_nostd': params_rmse})
        test_params_rmse_list.append(params_rmse)
        
        # Parameter_i RMSE 
        test_params_rmses_tmp = []
        for i in range(y.shape[1]):
            params_i_rmse = np.sqrt(mse(y_hat[:,i], y[:,i])) 
            trainer.logger.log_metrics({'test_params_{}_rmse_nostd'.format(i): params_i_rmse})
            test_params_rmses_tmp.append(params_i_rmse)
        test_params_rmses_tmp = np.array(test_params_rmses_tmp)
        test_params_rmses_list.append(test_params_rmses_tmp) 
        
        # Parameters MAE
        params_mae = mae(y_hat, y) 
        trainer.logger.log_metrics({'test_params_mae_nostd': params_mae})
        test_params_mae_list.append(params_mae)
        
        # Parameter_i MAE
        test_params_maes_tmp = []
        for i in range(y.shape[1]):
            params_i_mae = mae(y_hat[:,i], y[:,i]) 
            trainer.logger.log_metrics({'test_params_{}_mae_nostd'.format(i): params_i_mae})
            test_params_maes_tmp.append(params_i_mae)
        test_params_maes_tmp = np.array(test_params_maes_tmp)
        test_params_maes_list.append(test_params_maes_tmp) 
        
        #######################################################################
        #######################################################################

        #wandb.finish()
       
    params_overall_rmse     = np.mean(test_params_rmse_list)
    params_overall_rmse_std = np.std(test_params_rmse_list)/np.sqrt(num_runs_regressor)
    
    params_overall_mae      = np.mean(test_params_mae_list)
    params_overall_mae_std  = np.std(test_params_mae_list)/np.sqrt(num_runs_regressor)
    
    test_params_rmses_list  = np.array(test_params_rmses_list)
    params_rmse             = np.mean(test_params_rmses_list,axis=0)
    params_rmse_std         = np.std(test_params_rmses_list,axis=0)/np.sqrt(num_runs_regressor)
    
    test_params_maes_list  = np.array(test_params_maes_list)
    params_mae             = np.mean(test_params_maes_list,axis=0)
    params_mae_std         = np.std(test_params_maes_list,axis=0)/np.sqrt(num_runs_regressor)

    
    # write info about fit(s) to txt file
    if num_runs_regressor > 1:
        dictionary = repr({
                  'params_overall_rmse': params_overall_rmse,
                  'params_overall_rmse_std': params_overall_rmse_std,
                  'params_overall_mae': params_overall_mae,
                  'params_overall_mae_std': params_overall_mae_std,
                  'params_rmse': params_rmse,
                  'params_rmse_std': params_rmse_std,
                  'params_mae': params_mae,
                  'params_mae_std': params_mae_std,
                  'duration': time.time() - time1
                  })
    else:
        dictionary = repr({
                  'params_overall_rmse': params_overall_rmse,
                  'params_overall_mae': params_overall_mae,
                  'params_rmse': params_rmse,
                  'params_mae': params_mae,
                  'duration': time.time() - time1
                  })
    f = open( 'run_info.txt', 'a' )
    f.write( dictionary + '\n' )
    f.close()



#