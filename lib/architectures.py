#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Pytorch ANN architectures
"""
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F


class FC1(torch.nn.Module):
    '''
    ANN Architecture with 1 hidden, fully connected layer
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 drop_prob_1 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_1,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)

        x = self.fc_out(x)
        return x
    

class FC2(torch.nn.Module):
    '''
    ANN Architecture with 2 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_2,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)

        x = self.fc_out(x)
        return x
    

class FC3(torch.nn.Module):
    '''
    ANN Architecture with 3 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int,  
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_3,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)

        x = self.fc_out(x)
        return x
    

class FC4(torch.nn.Module):
    '''
    ANN Architecture with 4 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_4,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)

        x = self.fc_out(x)
        return x    

class FC5(torch.nn.Module):
    '''
    ANN Architecture with 5 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int,  
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 drop_prob_5 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.drop5 = nn.Dropout(p=drop_prob_5)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_5,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop5(x)

        x = self.fc_out(x)
        return x    

class FC6(torch.nn.Module):
    '''
    ANN Architecture with 6 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int, 
                 hidden_dim_6: int, 
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 drop_prob_5 = 0.0,
                 drop_prob_6 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.drop5 = nn.Dropout(p=drop_prob_5)
        
        self.fc6 = nn.Linear(in_features=hidden_dim_5,
                             out_features=hidden_dim_6)
        
        self.drop6 = nn.Dropout(p=drop_prob_6)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_6,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop5(x)
        
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.drop6(x)

        x = self.fc_out(x)
        return x 
    

class FC7(torch.nn.Module):
    '''
    ANN Architecture with 7 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int, 
                 hidden_dim_6: int,
                 hidden_dim_7: int, 
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 drop_prob_5 = 0.0,
                 drop_prob_6 = 0.0,
                 drop_prob_7 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.drop5 = nn.Dropout(p=drop_prob_5)
        
        self.fc6 = nn.Linear(in_features=hidden_dim_5,
                             out_features=hidden_dim_6)
        
        self.drop6 = nn.Dropout(p=drop_prob_6)
        
        
        self.fc7 = nn.Linear(in_features=hidden_dim_6,
                              out_features=hidden_dim_7)
        
        self.drop7 = nn.Dropout(p=drop_prob_7)
        
    
        self.fc_out = nn.Linear(in_features=hidden_dim_7,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop5(x)
        
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.drop7(x)

        x = self.fc_out(x)
        return x  

class FC8(torch.nn.Module):
    '''
    ANN Architecture with 8 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int, 
                 hidden_dim_6: int,
                 hidden_dim_7: int, 
                 hidden_dim_8: int, 
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 drop_prob_5 = 0.0,
                 drop_prob_6 = 0.0,
                 drop_prob_7 = 0.0,
                 drop_prob_8 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.drop5 = nn.Dropout(p=drop_prob_5)
        
        self.fc6 = nn.Linear(in_features=hidden_dim_5,
                             out_features=hidden_dim_6)
        
        self.drop6 = nn.Dropout(p=drop_prob_6)
        
        
        self.fc7 = nn.Linear(in_features=hidden_dim_6,
                              out_features=hidden_dim_7)
        
        self.drop7 = nn.Dropout(p=drop_prob_7)
        
        self.fc8 = nn.Linear(in_features=hidden_dim_7,
                             out_features=hidden_dim_8)
        
        self.drop8 = nn.Dropout(p=drop_prob_8)
        
    
        self.fc_out = nn.Linear(in_features=hidden_dim_8,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop5(x)
        
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.drop7(x)
        
        x = self.fc8(x)
        x = torch.relu(x)
        x = self.drop8(x)

        x = self.fc_out(x)
        return x 
    
class FC9(torch.nn.Module):
    '''
    ANN Architecture with 9 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int, 
                 hidden_dim_6: int,
                 hidden_dim_7: int, 
                 hidden_dim_8: int, 
                 hidden_dim_9: int, 
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 drop_prob_5 = 0.0,
                 drop_prob_6 = 0.0,
                 drop_prob_7 = 0.0,
                 drop_prob_8 = 0.0,
                 drop_prob_9 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.drop5 = nn.Dropout(p=drop_prob_5)
        
        self.fc6 = nn.Linear(in_features=hidden_dim_5,
                             out_features=hidden_dim_6)
        
        self.drop6 = nn.Dropout(p=drop_prob_6)
        
        
        self.fc7 = nn.Linear(in_features=hidden_dim_6,
                              out_features=hidden_dim_7)
        
        self.drop7 = nn.Dropout(p=drop_prob_7)
        
        self.fc8 = nn.Linear(in_features=hidden_dim_7,
                             out_features=hidden_dim_8)
        
        self.drop8 = nn.Dropout(p=drop_prob_8)
        
        self.fc9 = nn.Linear(in_features=hidden_dim_8,
                             out_features=hidden_dim_9)
        
        self.drop9 = nn.Dropout(p=drop_prob_9)
        
    
        self.fc_out = nn.Linear(in_features=hidden_dim_9,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop5(x)
        
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.drop7(x)
        
        x = self.fc8(x)
        x = torch.relu(x)
        x = self.drop8(x)
        
        x = self.fc9(x)
        x = torch.relu(x)
        x = self.drop9(x)

        x = self.fc_out(x)
        return x  


class FC10(torch.nn.Module):
    '''
    ANN Architecture with 10 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int, 
                 hidden_dim_6: int,
                 hidden_dim_7: int, 
                 hidden_dim_8: int, 
                 hidden_dim_9: int, 
                 hidden_dim_10: int,
                 drop_prob_1 = 0.0,
                 drop_prob_2 = 0.0,
                 drop_prob_3 = 0.0,
                 drop_prob_4 = 0.0,
                 drop_prob_5 = 0.0,
                 drop_prob_6 = 0.0,
                 drop_prob_7 = 0.0,
                 drop_prob_8 = 0.0,
                 drop_prob_9 = 0.0,
                 drop_prob_10 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.drop2 = nn.Dropout(p=drop_prob_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.drop3 = nn.Dropout(p=drop_prob_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.drop4 = nn.Dropout(p=drop_prob_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.drop5 = nn.Dropout(p=drop_prob_5)
        
        self.fc6 = nn.Linear(in_features=hidden_dim_5,
                             out_features=hidden_dim_6)
        
        self.drop6 = nn.Dropout(p=drop_prob_6)
        
        
        self.fc7 = nn.Linear(in_features=hidden_dim_6,
                              out_features=hidden_dim_7)
        
        self.drop7 = nn.Dropout(p=drop_prob_7)
        
        self.fc8 = nn.Linear(in_features=hidden_dim_7,
                             out_features=hidden_dim_8)
        
        self.drop8 = nn.Dropout(p=drop_prob_8)
        
        self.fc9 = nn.Linear(in_features=hidden_dim_8,
                             out_features=hidden_dim_9)
        
        self.drop9 = nn.Dropout(p=drop_prob_9)
        
        self.fc10 = nn.Linear(in_features=hidden_dim_9,
                             out_features=hidden_dim_10)
        
        self.drop10 = nn.Dropout(p=drop_prob_10)
        
    
        self.fc_out = nn.Linear(in_features=hidden_dim_10,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop5(x)
        
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.drop7(x)
        
        x = self.fc8(x)
        x = torch.relu(x)
        x = self.drop8(x)
        
        x = self.fc9(x)
        x = torch.relu(x)
        x = self.drop9(x)
        
        x = self.fc10(x)
        x = torch.relu(x)
        x = self.drop10(x)

        x = self.fc_out(x)
        return x  
    

class Conv3FC1(torch.nn.Module):
    '''
    ANN Architecture with 2 hidden convolutional layers 
    plus one fully connected layer
    
    Uses two successive input functions as two different channels for first 
    Conv layer
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Convolution layers
                 conv_channels_1a: int = 16, 
                 conv_channels_1b: int = 16, 
                 kernel_len_1: int = 5,
                 
                 conv_channels_2: int = 16, 
                 conv_channels_2b: int = 16, 
                 kernel_len_2: int = 5,
                 
                 ### Fully connected layers
                 hidden_dim_1: int = 32, 
                 # drop_prob_1 = 0.0,
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()
        
        wandb.config.update({"conv_channels_1a": conv_channels_1a, "conv_channels_1b": conv_channels_1b, 
                             "kernel_len_1": kernel_len_1, 
                             "conv_channels_2": conv_channels_2, "kernel_len_2": kernel_len_2, 
                             "hidden_dim_1": hidden_dim_1})
        
        self.conv1a = nn.Conv1d(2, conv_channels_1a, kernel_len_1, stride=1, padding=0)
        
        self.conv1b = nn.Conv1d(conv_channels_1a, conv_channels_1b, kernel_len_1, stride=1, padding=0)
        
        self.pool1  = nn.AvgPool1d(3, stride=3)

        self.conv2  = nn.Conv1d(conv_channels_1b, conv_channels_2, kernel_len_2, stride=1, padding=0)

        self.conv2b  = nn.Conv1d(conv_channels_2, conv_channels_2b, kernel_len_2, stride=1, padding=0)

        self.pool2  = nn.AvgPool1d(3, stride=3)  
        self.pool2b  = nn.AvgPool1d(5, stride=5)  
        
        
        
        x = torch.randn(1000,2,64).view(-1,2,64)
        self.to_linear = None
        self.convs3(x)
        
        
        self.fc1    = nn.Linear(in_features=self.to_linear,
                              out_features=hidden_dim_1)
        
        # self.drop1 = nn.Dropout(p=drop_prob_1)
        
        self.fc_out = nn.Linear(in_features=hidden_dim_1,
                              out_features=out_features)




    def convs1(self, x):
        # for kernel size 3 + 3

        x = F.relu(self.conv1a(x))  # data is now: 16x62
        x = F.relu(self.conv1b(x))  # data is now: 16x60 
        
        x = self.pool1(x)     # data is now: 16x20 
        
        x = F.relu(self.conv2(x))   # data is now: 32x18 
        
        x = self.pool2(x)     # data is now: 32x6

        if self.to_linear is None:
            print("shape total: ",  x.shape)
            print("shape first element in batch: ", x[0].shape)
            self.to_linear = x[0].shape[0] * x[0].shape[1]
            print("self.to_linear: ", self.to_linear)
        return x
        
        
    def convs2(self, x):
        # for kernel size 5 + 7

        x = F.relu(self.conv1a(x))  # data is now: 16x60
        x = F.relu(self.conv1b(x))  # data is now: 16x54 
        
        x = self.pool1(x)     # data is now: 16x18 
        
        x = F.relu(self.conv2(x))   # data is now: 32x12 
        
        x = self.pool2(x)     # data is now: 32x4

        if self.to_linear is None:
            print("shape total: ",  x.shape)
            print("shape first element in batch: ", x[0].shape)
            self.to_linear = x[0].shape[0] * x[0].shape[1]
            print("self.to_linear: ", self.to_linear)
        return x
        
        
    def convs3(self, x):
        # for kernel size 5 + 5

        x = F.relu(self.conv1a(x))  # data is now: 16x60
        x = F.relu(self.conv1b(x))  # data is now: 16x54 
        
        x = self.pool1(x)     # data is now: 16x18 
        
        x = F.relu(self.conv2(x))   # data is now: 32x14 
        x = F.relu(self.conv2b(x))   # data is now: 32x10 
        
        x = self.pool2b(x)     # data is now: 32x2

        if self.to_linear is None:
            print("shape total: ",  x.shape)
            print("shape first element in batch: ", x[0].shape)
            self.to_linear = x[0].shape[0] * x[0].shape[1]
            print("self.to_linear: ", self.to_linear)
        return x
        
        
    def forward(self, x):
        
        batch_size_here = x.size(dim=0)
        
        x = torch.reshape(x, (batch_size_here, 2, -1))  # data is now: 2x64  (channels x points, batch dim omitted)
        
        x = self.convs3(x)
        
        x = torch.flatten(x, 1)  # flatten everything but the batch dimension
                                 # data is now: 192 
        
        x = self.fc1(x)         # data is now: 16
        x = F.relu(x)
        
        x = self.fc_out(x)      # data is now: 9

        return x
    
    
    
    
    
    
    
    
    
    
    
    
    #