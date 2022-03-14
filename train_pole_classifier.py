#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:44:41 2020

@authors: andreaskrassnigg, siegfriedkaidisch

Classifier based on pytorch basic template: Training file

"""
from pytorch_lightning import loggers as pl_loggers
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import wandb
import argparse
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import matplotlib.pyplot as plt

from lib.pole_classifier import Pole_Classifier, PoleDataModule_Classifier
from lib.plotting_routines import classifier_plot

from parameters import train_portion_classifier, val_portion_classifier, test_portion_classifier
from parameters import architecture_classifier
from parameters import hidden_dim_1_classifier, hidden_dim_2_classifier, hidden_dim_3_classifier
from parameters import hidden_dim_4_classifier, hidden_dim_5_classifier, hidden_dim_6_classifier
from parameters import in_features_classifier, out_features_classifier
from parameters import weight_decay_classifier, batch_size_classifier, learning_rate_classifier, epochs_classifier
from parameters import data_dir_classifier, log_dir_classifier, models_dir_classifier
from parameters import n_examples_classifier
from parameters import re_max, re_min, im_max, im_min, coeff_re_max, coeff_re_min, coeff_im_max, coeff_im_min
from parameters import num_runs_classifier
from parameters import optimizer_classifier
from parameters import drop_prob_1_classifier, drop_prob_2_classifier, drop_prob_3_classifier
from parameters import drop_prob_4_classifier, drop_prob_5_classifier, drop_prob_6_classifier
from parameters import val_check_interval_classifier, es_patience_classifier
from parameters import input_name_classifier   

##############################################################################
##########################   Execution   #####################################
##############################################################################


if __name__ == '__main__':
    ##########   Parse hyperparameters, if wanted   ##########################
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_check_interval_classifier', type=float, default=val_check_interval_classifier)
    parser.add_argument('--es_patience_classifier', type=int, default=es_patience_classifier)  
    args = parser.parse_args()
    val_check_interval_classifier = args.val_check_interval_classifier
    es_patience_classifier = args.es_patience_classifier
    '''
    ##########################################################################
    time1 = time.time()
    seed_everything(seed=1234)  #standard: 1234
    seeds = np.random.randint(0,1e6,size=[num_runs_classifier]) 
    
    net_hyperparameters = dict(
                architecture = architecture_classifier,
                in_features  = in_features_classifier,
                out_features = out_features_classifier, 
                hidden_dim_1 = hidden_dim_1_classifier,
                hidden_dim_2 = hidden_dim_2_classifier,
                hidden_dim_3 = hidden_dim_3_classifier,
                hidden_dim_4 = hidden_dim_4_classifier,
                hidden_dim_5 = hidden_dim_5_classifier,
                hidden_dim_6 = hidden_dim_6_classifier,
                
                weight_decay  = weight_decay_classifier,  
                drop_prob_1   = drop_prob_1_classifier,
                drop_prob_2   = drop_prob_2_classifier,
                drop_prob_3   = drop_prob_3_classifier,
                drop_prob_4   = drop_prob_4_classifier,
                drop_prob_5   = drop_prob_5_classifier,
                drop_prob_6   = drop_prob_6_classifier,
                
                learning_rate = learning_rate_classifier,
                optimizer     = optimizer_classifier
		)
    
    other_hyperparameters = dict(
                re_max = re_max, 
                re_min = re_min, 
                im_max = im_max, 
                im_min = im_min, 
                coeff_re_max = coeff_re_max, 
                coeff_re_min = coeff_re_min, 
                coeff_im_max = coeff_im_max, 
                coeff_im_min = coeff_im_min,
                
                n_examples_classifier    = n_examples_classifier,
                train_portion_classifier = train_portion_classifier,
                val_portion_classifier   = val_portion_classifier,
                test_portion_classifier  = test_portion_classifier, 
                batch_size               = batch_size_classifier,
                
                val_check_interval = val_check_interval_classifier,
                es_patience        = es_patience_classifier,

                input = input_name_classifier
		)
    
    hyperparameters = {**net_hyperparameters, **other_hyperparameters}
    
    test_accs   = []
    test_losses = []
    for i in range(num_runs_classifier): # average test_acc over multiple runs, if num_runs>1
        seed_everything(seed=seeds[i])
        name   = 'classifier_run_' + str(time.time())
    
        #wandb.init(config=hyperparameters,
        #       entity="ml-tpp", project="pole_classifier",
        #       group="",
        #       notes="",
        #       tags = ["Classifier"])

        #logger = WandbLogger(save_dir=log_dir_regressor, name=name) 
        logger = TensorBoardLogger(log_dir_classifier, name)
    
        model = Pole_Classifier(
                    **net_hyperparameters
                    )
                    
        datamodule = PoleDataModule_Classifier(data_dir=data_dir_classifier, batch_size=batch_size_classifier, 
                                    train_portion=train_portion_classifier, validation_portion=val_portion_classifier, test_portion=test_portion_classifier)
        
        checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
            dirpath = models_dir_classifier,
            #filename = 'name_' + str(wandb.run.name) + '_id_' + str(wandb.run.id) + '_' + str(i),
            filename = name,
            monitor="val_acc",
            save_top_k=1, 
            mode='max',
            save_last= False
        )
        
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=es_patience_classifier, mode="max")
        
        trainer = pl.Trainer(
            logger=logger,
            val_check_interval=val_check_interval_classifier,
            callbacks=[checkpoint_callback1, early_stop_callback],
            max_epochs=epochs_classifier,
            gpus=1
        )
    
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model=model, datamodule=datamodule)
        
        # Manually do test step     
        test_data_X = torch.from_numpy(datamodule.all_data.test_data_X)
        test_data_Y = datamodule.all_data.test_data_Y.reshape(-1,1)
        model = Pole_Classifier.load_from_checkpoint(checkpoint_path=models_dir_classifier+name+'.ckpt')
        model.eval()
        model.to(device='cpu')
        preds = (torch.argmax(model(test_data_X), dim=1)).cpu().detach().numpy().reshape(-1,1)
        test_acc = np.mean(preds==test_data_Y)
        trainer.logger.log_metrics({'test_acc': test_acc})
        print('-------------------------------------------')
        print('Test Accuracy: ', test_acc)
        print('-------------------------------------------')
        
        test_accs.append(test_acc)
        #wandb.finish()
        
    test_acc_mean        = np.mean(test_accs)
    test_acc_mean_error  = np.std(test_accs)/np.sqrt(num_runs_classifier)
    
    # write info about fit(s) to txt file
    if num_runs_classifier > 1:
        dictionary = repr({
                  'test_acc_mean': test_acc_mean,
                  'test_acc_mean_error': test_acc_mean_error,
                  'duration': time.time() - time1
                  })
    else:
        dictionary = repr({
                  'test_acc_mean': test_acc_mean,
                  'duration': time.time() - time1
                  })
    f = open( 'run_info.txt', 'a' )
    f.write( dictionary + '\n' )
    f.close()

    # Create classifier plot with (latest) test data
    fig, _ = classifier_plot(labels=test_data_Y, predictions=preds, do_return=True)
    plt.show()
    

###############################################################################
###############################################################################
###############################################################################




#
