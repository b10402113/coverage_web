#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load model.py
import os
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import json

#from keras.utils import to_categorical
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from functools import partial
from sklearn.utils import shuffle


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device

batch_size = 256
LL = ['LONGITUDE', 'LATITUDE']
image_dir = '../results/image/'
json_dir = '../results/json/'

class Dataset(Dataset):
    def __init__(self, df, train):
        self.df = df
        self.features = self.df[LL+['LONGITUDE_SOURCE', 'LATITUDE_SOURCE']].values
        self.train = train
        
        if train :
            self.labels = self.df['DIFF_SIGNAL'].values 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.train:
            return self.features[idx,:2], self.features[idx,2:], self.labels[idx]
        else:
            return self.features[idx,:2], self.features[idx,2:]

def get_loader(df_train, df_val) :
    train_dataset = Dataset(df_train, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = Dataset(df_val, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader

# https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.Dropout(0.2),  #0.2
            
            nn.Linear(256, 128),####
            nn.ReLU(True),
            nn.Linear(128, 64),####
            nn.ReLU(True),
            nn.Linear(64, 32),####
            nn.ReLU(True),
            nn.Linear(32, 3))
        
        '''self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.Dropout(0.2),  #0.2
            
            nn.Linear(128, 64),####
            nn.ReLU(True),
            nn.Linear(64, 32),####
            nn.ReLU(True),
            nn.Linear(32, 3))'''
        
        '''self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.Dropout(0.2),  #0.2
            
            nn.Linear(32, 64),####
            nn.ReLU(True),
            nn.Linear(64, 32),####
            nn.ReLU(True),
            nn.Linear(32, 3))'''

    def forward(self, x1, x2=None, predict=False):
        out1 = self.encoder(x1)
        
        if predict :
            return out1
        
        out2 = self.encoder(x2)
        distance = torch.sqrt(torch.sum((out1 - out2)**2, dim=1))
        return distance
    
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=5):
    if epoch % lr_decay_epoch:
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr'] / (1+lr_decay*epoch)
            param_group['lr'] = new_lr
    return optimizer

def get_model(num_epochs=150, lr=1e-1, weight_decay=1e-5, lr_decay=1e-1, lr_decay_epoch=5):
    model = autoencoder().to(device).float()
    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                                weight_decay=weight_decay)
    optimizer = exp_lr_scheduler(optimizer, num_epochs, 
                                 lr_decay=lr_decay, lr_decay_epoch=lr_decay_epoch)
    return model, criterion, optimizer

def train(model, criterion, optimizer, train_dataloader, num_epochs=150, show=False, model_info = None):
    losses = []
    min_loss = 1
    best_model = model
    for epoch in range(num_epochs):
        for features1, features2, labels in train_dataloader:
            features1 = features1.to(device)
            features2 = features2.to(device)
            labels = labels.to(device)
            # ===================forward=====================
            distance = model(features1.float(), features2.float())
            loss = criterion(distance, labels.float())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        losses.append(loss.item())
        
        if losses[-1] < min_loss :
            min_loss = losses[-1]
            best_model = copy.deepcopy(model) 

    best_model = best_model.eval()
    
    if show:
        if type(model_info) == int :
            #print("Set", s, 'model loss start {:.4f} best {:.4f}'.format(losses[0], min_loss))
            #####-----PRINT JSON-----#####
            json_data = {"Set": model_info, 'model loss start':losses[0], 'best': min_loss}
            json_data = json.dumps(json_data)
            with open(json_dir+'training_info.json', 'w') as f:
                f.write(json_data)
                
        elif type(model_info) == tuple :
            json_data = {"B": model_info[0], "F":model_info[1], 'model loss start':losses[0], 'best': min_loss}
            json_data = json.dumps(json_data)
            with open(json_dir+'training_info.json', 'w') as f:
                f.write(json_data)
        
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('MSE', fontsize=20)
        plt.tick_params(labelsize=15)
        plt.title('MSE loss in Training Phase', fontsize=25)
        plt.grid(True)
        plt.savefig(image_dir+'training_MSE.png')
        plt.close()
        #plt.show()
    
    return best_model

def evaluate(best_model, val_data) :
    val_dataset = Dataset(val_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                shuffle=False)
    
    distance = np.empty(0)
    for features1, features2 in val_dataloader:
        features1 = features1.to(device)
        features2 = features2.to(device)
        out = best_model(features1.float(), features2.float())
        out = out.cpu().detach().numpy()
        distance = np.append(distance, out, axis=0)
        
    return distance


# In[ ]:




