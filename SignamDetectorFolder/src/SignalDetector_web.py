#!/usr/bin/env python
# coding: utf-8

# In[2]:

####
#import function_set.loadnotebook

import os
import math
import copy
import traceback
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.cm as cm
import pickle

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

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from xgboost import XGBRFRegressor

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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

from function_set.helper import * 
from model.model import *
#from model.model_original import *

import time
start = time.clock()


# In[3]:


class SignalDetector:
    def get_dataset_source(self, source):
        if source == 'UJIdataset':
            self.source = source
            self.data_dir = '../data/UJIIndoorLoc/restructured_train.csv'
        elif source == 'ITRI-indoor':
            self.source = source
            self.data_dir = '../data/ITRI-indoor/restructed_csv/all_summary.csv'
        elif source == 'ITRI-outdoor':
            self.source = source
            self.data_dir = '../data/ITRI-outdoor/restructed_csv/restructed.csv'
        else:
            print('- Please input \'UJIdataset\', \'ITRI-indoor\' or \'ITRI-outdoor\' and other dataset are not surppoted yet. -')
    def __init__(self, source):
        #define source
        self.get_dataset_source(source)
        
        #read csv data(have to restruct first)
        self.df = pd.read_csv(self.data_dir)

        #turn coodinate to range (-1, 1)
        self.df['LONGITUDE'] = minmax_scale(self.df['LONGITUDE'], feature_range=(-1, 1))
        self.df['LATITUDE'] = minmax_scale(self.df['LATITUDE'], feature_range=(-1, 1))

        #Here, LL means Longitude and Latidue, S means set, W means WAP. This is used to choose certain collumns of df
        if source == 'UJIdataset':
            filter_ = LL + ['BUILDINGID', 'FLOOR', 'WAP']
        elif source == 'ITRI-indoor':
            filter_ = LL + ['SET', 'WAP']
        else:
            filter_ = LL + ['WAP']

        #replace signal 100 as -200
        self.df['SIGNAL'].replace(100, -200, inplace=True)
        #turn signal to range (-1,1)
        self.df['SIGNAL'] = minmax_scale(self.df['SIGNAL'], feature_range=(-1, 1))
        #mark out the minimum signal at each location
        self.df_agg_max_all_data = merge_agg(self.df, filter_, 'SIGNAL', ['min'], ['MIN_SIGNAL'])
        
    def set_model_info(self, model_info):
        if self.source == ('ITRI-indoor' or 'UJIdataset'):
            self.model_info = model_info
        else:
            print('- Model_info is not supported in this dataset -')
            self.model_info = None
        
    def encoder_result(self, lr=1e-1, lr_decay=1e-1, 
                   lr_decay_epoch=5, weight_decay=1e-5, show=False, model_info = None) :
    
        train_dataloader, val_dataloader = get_loader(self.df_train, self.df_val)

        model, criterion, optimizer = get_model(num_epochs=100, 
                                                lr=lr, lr_decay=lr_decay,
                                                weight_decay=weight_decay,  
                                                lr_decay_epoch=int(lr_decay_epoch))
        
        best_model = train(model, criterion, optimizer, 
                           train_dataloader,
                           num_epochs=100, #100 
                           show=show, model_info=model_info)
            

        #signal LLW
        distance = evaluate(best_model, self.df_val)
        
        mse = mean_squared_error(self.df_val['DIFF_SIGNAL'], distance)

        if not show :
            return -mse

        train_distance = evaluate(best_model, self.df_train)
        detected = self.df_train[(self.df_train.SIGNAL>-1)].index.values
        max_distance = 0.1*train_distance[detected].max()

        # coverage LLW
        covered = distance<max_distance
        c_acc = np.mean((self.df_val['SIGNAL'] > -1) == covered)

        df_llw = self.df_val.copy()
        df_llw['distance'] = distance
        df_llw['covered'] = covered

        generated_distance = evaluate(best_model, self.val_generated)
        self.val_generated['pred_COVERAGE'] = (generated_distance<max_distance).astype(int)
        generated_coverage = self.val_generated.groupby(LL)['pred_COVERAGE'].sum().reset_index()

        # coverage LL
        df_ll = self.df_val_per_locs.merge(generated_coverage, on=LL)
        c_mse = mean_squared_error(df_ll['COVERAGE'], df_ll['pred_COVERAGE'])

        if self.pollute_happened:
            # pilot pollution LL
            df_ll['pred_POLLUTED'] = df_ll['pred_COVERAGE'] > 1
            p_acc = np.mean(((df_ll['STATUS']==2) == df_ll['pred_POLLUTED']))

        # hole LL      
        df_ll['pred_HOLE'] = df_ll['pred_COVERAGE'] == 0
        h_acc = np.mean(((df_ll['STATUS']==0) == df_ll['pred_HOLE']))

        if self.pollute_happened:
            #todo coverage mse and coverage acc
            #print('encoder', 'mse signal LLW', mse, 'coverage LLW', c_acc) 
            #print('coverage LL', c_mse, 'acc polluted LL', p_acc, 'hole LL', h_acc)
            #####-----PRINT JSON-----#####
            json_data = {'encoder': {'mse signal LLW': mse, 
                                     'coverage LLW': c_acc,
                                    'coverage LL': c_mse,
                                     'acc polluted LL': p_acc,
                                     'hole LL': h_acc}}
            json_data = json.dumps(json_data)
            with open(json_dir+'AE.json', 'w') as f:
                f.write(json_data)
            
            
        else:
            #print('encoder', 'mse signal LLW', mse, 'coverage LLW', c_acc) 
            #print('coverage LL', c_mse, 'hole LL', h_acc)
            #####-----PRINT JSON-----#####
            json_data = {'encoder': {'mse signal LLW': mse, 
                                     'coverage LLW': c_acc,
                                    'coverage LL': c_mse,
                                     'hole LL': h_acc}}
            json_data = json.dumps(json_data)
            with open(json_dir+'AE.json', 'w') as f:
                f.write(json_data)

        dir = "..\\results\\model"
        if type(self.model_info) == int:
            torch.save(best_model.state_dict(),os.path.join(dir,"AE_set" + str(self.model_info) +                                                            "_model.pth"))
        elif type(self.model_info) ==tuple:
            torch.save(best_model.state_dict(),os.path.join(dir,"AE_b" + str(self.model_info[0]) +                                                            'f' + str(self.model_info[1]) +                                                            "_model.pth"))
        else:
            torch.save(best_model.state_dict(),os.path.join(dir,"AE_outdoor_model.pth"))
        print("- Model has been SAVED! -")
        self.enc_preds = df_llw, df_ll
        return df_llw, df_ll

    def optimize_encoder(self):
        encoder_fn = partial(self.encoder_result)

        optimizer = BayesianOptimization(
            f=encoder_fn,
            
             pbounds={"lr": (1e-4, 5e-2), 
                 "lr_decay": (1e-5, 5e-4), 
                 "lr_decay_epoch": (5, 15), 
                 "weight_decay": (1e-5, 5e-4)}, #5e-1
            random_state=100,
            verbose=2
        )
        optimizer.maximize(n_iter=10, init_points=5)
        return optimizer
    
    def get_data_train_val(self):
        try:
            #filter out the set data,         
            data = get_train_val(self.model_info, self.df_agg_max_all_data)
            self.df_train_per_locs, self.df_val_per_locs, self.df_train, self.df_val, self.val_generated, self.pollute_happened = data
            
            
            plot_data(self.df_train_per_locs, self.df_val_per_locs)
        except Exception :
            traceback.print_exc()
            
    def train_AE(self):
        try:
            self.opt = self.optimize_encoder()
            #|     |     |      |    |
            
            #self.enc_preds = self.encoder_result(show=True, lr=0.5124, lr_decay=0.2965, 
             #                                   lr_decay_epoch=85.42, weight_decay=0.04779,
              #                                   model_info=self.model_info)
            self.enc_preds = self.encoder_result(show=True, **self.opt.max['params'],
                                                 model_info=self.model_info)
            
        except Exception :
            traceback.print_exc()
    def train_KNN(self):
        try:
            self.knn_signal = KNeighborsRegressor(n_neighbors=4)
            self.knn_coverage = KNeighborsClassifier(n_neighbors=4)
            self.knn_coverage2 = KNeighborsRegressor(n_neighbors=4)
            self.knn_polluted = KNeighborsClassifier(n_neighbors=4)
            self.knn_hole = KNeighborsClassifier(n_neighbors=4)
            
            self.knn_preds = other_result('knn', self.knn_signal, 
                                     self.knn_coverage, self.knn_coverage2, 
                                     self.knn_polluted, self.knn_hole, 
                                     self.df_train, self.df_val, 
                                     self.df_train_per_locs, self.df_val_per_locs,
                                     self.pollute_happened, self.model_info)
        except Exception :
            traceback.print_exc()
            
    def train_SVM(self):
        try:
            self.svm_signal = SVR(gamma='scale')
            self.svm_coverage = SVC(gamma='scale')
            self.svm_coverage2 = SVR(gamma='scale')
            self.svm_polluted = SVC(gamma='scale')
            self.svm_hole = SVC(gamma='scale')
            self.svm_preds = other_result('svm', self.svm_signal, 
                                     self.svm_coverage, self.svm_coverage2, 
                                     self.svm_polluted, self.svm_hole, 
                                     self.df_train, self.df_val, 
                                     self.df_train_per_locs, self.df_val_per_locs,
                                          self.pollute_happened, self.model_info)
        except Exception :
            traceback.print_exc()
            
    def train_DT(self):
        try:
            self.dt_signal = DecisionTreeRegressor()
            self.dt_coverage = DecisionTreeClassifier()
            self.dt_coverage2 = DecisionTreeClassifier()
            self.dt_polluted = DecisionTreeClassifier()
            self.dt_hole = DecisionTreeClassifier()
            self.dt_preds = other_result('dt', self.dt_signal, 
                                     self.dt_coverage, self.dt_coverage2, 
                                     self.dt_polluted, self.dt_hole, 
                                     self.df_train, self.df_val, 
                                     self.df_train_per_locs, self.df_val_per_locs,
                                         self.pollute_happened, self.model_info)
        except Exception :
            traceback.print_exc()

    def train_XGB(self):
        try:
            self.xgb_signal = XGBRegressor()
            self.xgb_coverage = XGBClassifier()
            self.xgb_coverage2 = XGBRFClassifier()
            self.xgb_polluted = XGBClassifier()
            self.xgb_hole = XGBClassifier()
            self.xgb_preds = other_result('xgb', self.xgb_signal, 
                                     self.xgb_coverage, self.xgb_coverage2, 
                                     self.xgb_polluted, self.xgb_hole, 
                                     self.df_train, self.df_val, 
                                     self.df_train_per_locs, self.df_val_per_locs,
                                         self.pollute_happened, self.model_info)
        except Exception :
            traceback.print_exc()
    
    def train_LGBM(self):
        try:
            import lightgbm

            #categorical_features = [c for c, LL+['WAP']]
            categorical_features = [c for c, col in enumerate(MySD.df_train.columns) if 'cat' in col]
            x_train = MySD.df_train[LL+['WAP']]
            x_val = MySD.df_val[LL+['WAP']]

            train_data = lightgbm.Dataset(x_train, label=MySD.df_train['SIGNAL'], categorical_feature=categorical_features)
            test_data = lightgbm.Dataset(x_val, label=MySD.df_val['SIGNAL'])

            parameters = {
                'application': 'binary',
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance': 'true',
                'boosting': 'gbdt',
                'num_leaves': 31,
                'feature_fraction': 0.5,
                'bagging_fraction': 0.5,
                'bagging_freq': 20,
                'learning_rate': 0.05,
                'verbose': 0
            }

            model = lightgbm.train(parameters,
                                   train_data,
                                   valid_sets=test_data,
                                   num_boost_round=5000,
                                   early_stopping_rounds=100)
            
        except Exception :
            traceback.print_exc()
        
    def plot_result(self):
        if not hasattr(self, 'df_val_per_locs'):
            print('- Plese do get_data_train_vals(). -')
        elif not hasattr(self, 'enc_preds'):
            print('- Plese do train_AE(). -')
        elif not hasattr(self, 'knn_preds'):
            print('- Plese do train_KNN(). -')
        elif not hasattr(self, 'svm_preds'):
            print('- Plese do train_SVM(). -')
        elif not hasattr(self, 'dt_preds'):
            print('- Plese do train_DT(). -')
        elif not hasattr(self, 'xgb_preds'):
            print('- Plese do train_XGB(). -')
            
        else:
            try:
                if self.pollute_happened:
                    for t in [ 'coverage', 'polluted', 'hole'] :
                        fig = plt.figure(figsize=(25,15))
                        plot_encoder(1, self.enc_preds, plot_type=t)
                        plot_other(2, self.df_val_per_locs, self.knn_preds, 'knn', self.pollute_happened, plot_type=t)
                        plot_other(3, self.df_val_per_locs, self.svm_preds, 'svm', self.pollute_happened, plot_type=t)
                        plot_other(4, self.df_val_per_locs, self.dt_preds, 'dt', self.pollute_happened, plot_type=t)
                        plot_other(5, self.df_val_per_locs, self.xgb_preds, 'xgb', self.pollute_happened, plot_type=t)
                        
                        plt.show()

                    _, df_ll = self.enc_preds
                    y_true = df_ll['STATUS']==2
                    y_pred_enc = df_ll['pred_POLLUTED']
                    y_pred_knn = self.knn_preds[3]
                    y_pred_dt = self.dt_preds[3]
                    y_pred_svm = self.svm_preds[3]
                    y_pred_xgb = self.xgb_preds[3]

                    print("-------------------------Pollution-----------------------------")
                    print("AE F1-score: ",f1_score(y_true, y_pred_enc))
                    print("KNN F1-score: ",f1_score(y_true, y_pred_knn))
                    print("DT F1-score: ",f1_score(y_true, y_pred_dt))
                    print("SVM F1-score: ",f1_score(y_true, y_pred_svm))
                    print("XGB F1-score: ",f1_score(y_true, y_pred_xgb))
                    print("--------------------------Pollution----------------------------")
                    print("AE recall score: ", recall_score(y_true, y_pred_enc))
                    print("KNN recall score: ", recall_score(y_true, y_pred_knn))
                    print("DT recall score: ", recall_score(y_true, y_pred_dt))
                    print("SVM recall score: ", recall_score(y_true, y_pred_svm))
                    print("XGB recall score: ", recall_score(y_true, y_pred_xgb))
                    print("-----------------------Pollution-------------------------------")
                else:
                    for t in [ 'coverage', 'hole'] :
                        fig = plt.figure(figsize=(25,15))
                        plot_encoder(1, self.enc_preds, plot_type=t)
                        plot_other(2, self.df_val_per_locs, self.knn_preds, 'knn', self.pollute_happened, plot_type=t)
                        plot_other(3, self.df_val_per_locs, self.svm_preds, 'svm', self.pollute_happened, plot_type=t)
                        plot_other(4, self.df_val_per_locs, self.dt_preds, 'dt', self.pollute_happened, plot_type=t)
                        plot_other(5, self.df_val_per_locs, self.xgb_preds, 'dt', self.pollute_happened, plot_type=t)
                        plt.show()

                _, df_ll = self.enc_preds
                y_true = df_ll['STATUS']==0
                y_pred_enc = df_ll['pred_HOLE']
                if self.pollute_happened:
                    y_pred_knn = self.knn_preds[4]
                    y_pred_dt = self.dt_preds[4]
                    y_pred_svm = self.svm_preds[4]
                    y_pred_xgb = self.xgb_preds[4]
                else:
                    y_pred_knn = self.knn_preds[3]
                    y_pred_dt = self.dt_preds[3]
                    y_pred_svm = self.svm_preds[3]
                    y_pred_xgb = self.xgb_preds[3]

                print("-------------------------HOLE-----------------------------")
                print("AE F1-score: ",f1_score(y_true, y_pred_enc))
                print("KNN F1-score: ",f1_score(y_true, y_pred_knn))
                print("DT F1-score: ",f1_score(y_true, y_pred_dt))
                print("SVM F1-score: ",f1_score(y_true, y_pred_svm))
                print("XGB F1-score: ",f1_score(y_true, y_pred_xgb))
                print("--------------------------HOLE----------------------------")
                print("AE recall score: ", recall_score(y_true, y_pred_enc))
                print("KNN recall score: ", recall_score(y_true, y_pred_knn))
                print("DT recall score: ", recall_score(y_true, y_pred_dt))
                print("SVM recall score: ", recall_score(y_true, y_pred_svm))
                print("XGB recall score: ", recall_score(y_true, y_pred_xgb))
                print("-----------------------HOLE-------------------------------")
            except Exception :
                traceback.print_exc()
            
    def do_experiment(self) :
        self.get_data_train_val()
        self.train_AE()
        self.train_KNN()
        self.train_SVM()
        self.train_DT()
        self.train_XGB()
        self.plot_result()
        
'''if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='ITRI-indoor', help = '[UJIIndoorLoc, ITRI-indoor, ITRI-outdoor]')  
    parser.add_argument('--power-config-set', type=int, default=16, help = 'if use ITRI-indoor dataset, this can be 1 ~ 33.')
    
    parser.add_argument('--abcdefg', type=int, help='blablablablablabla')
    
    
    test = parser.parse_args(args = [])'''
def main(data_source = 'ITRI-indoor', model_info = 1):
    MySD = SignalDetector(data_source)
    MySD.set_model_info(model_info)
    MySD.get_data_train_val()
    try:
        MySD.train_AE()
    except Exception:
        MySD.train_AE()
    MySD.train_KNN()
    MySD.train_SVM()
    MySD.train_DT()
    MySD.train_XGB()
    MySD.plot_result()


# In[4]:





# In[ ]:




