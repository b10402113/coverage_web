#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load helper.py
import os
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.externals import joblib 

from keras.utils import to_categorical
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

import json
absFilePath = os.path.abspath(__file__)
print('absFilePath'+absFilePath)
fileDir = os.path.dirname(os.path.abspath(__file__))
print('fileDir'+fileDir)
parentDir = os.path.dirname(fileDir)
parentDir = os.path.dirname(parentDir)
print('parentDir'+parentDir)
json_dir = parentDir+"/results/json/"
image_dir = parentDir+"/results/image/"


LL = ['LONGITUDE', 'LATITUDE']
LLS = LL + ['SET']
LLW = LL + ['WAP']
LLSW = LLW + ['SET']

def df_filter(df, s) :
    filtered_df = df.loc[df['SET']==s, :].copy()
    filtered_df.drop(['SET'],axis=1, inplace=True)
    return filtered_df

def get_hole(not_detected, WAP_number):
    hole = not_detected.groupby(LL).count().reset_index()[LLW]
    hole = hole[hole['WAP'] == WAP_number]
    
    hole = hole[LL].drop_duplicates()
    hole['STATUS'] = 0 
    hole['COVERAGE'] = 0 
    hole = hole.set_index(LL)
    return hole

def get_poll(detected) :
    poll = detected.groupby(LL).agg(['mean', 'count']).dropna().reset_index()
    poll.columns = ['_'.join(tup).rstrip('_') for tup in poll.columns.values]
    
    poll = poll.rename({'WAP_count':'COVERAGE'}, axis=1)
    poll['STATUS'] = 1
    
    threshold_poll = np.percentile(poll.SIGNAL_mean, 75) #origin is 75
    poll['POLLUTION'] = poll.COVERAGE > 1# & (poll.SIGNAL_mean > threshold_poll)
    poll.loc[poll.POLLUTION, 'STATUS'] = 2
    
    poll = poll[LL+['COVERAGE', 'STATUS']].drop_duplicates()
    poll = poll.set_index(LL)
    return poll

def get_train_val(s, df_agg_max_all_set, r=0) :
    df_agg_max = df_agg_max_all_set.loc[df_agg_max_all_set['SET']==s, :].copy()
    detected_filter = df_agg_max['MIN_SIGNAL']>-1
    # llw stands for longitude, Latitude, WAP
    # means the locations are always covered by current WAP 
    df_detected_llw = df_agg_max[detected_filter].drop('MIN_SIGNAL', axis=1)
    df_detected_llw = merge_agg(df_detected_llw, LLSW, 'SIGNAL', ['mean'])
    df_detected_llw = df_detected_llw.drop('SIGNAL', axis=1).rename(columns={'mean' : 'SIGNAL'})
    df_detected_llw = df_detected_llw.drop_duplicates().reset_index(drop=True)
    
    # means the locations are not covered (once or more) by current WAP 
    df_not_detected_llw = df_agg_max[~detected_filter]
    df_not_detected_llw = df_not_detected_llw.drop('SIGNAL', axis=1)
    df_not_detected_llw = df_not_detected_llw.rename(columns={'MIN_SIGNAL' : 'SIGNAL'})
    df_not_detected_llw = df_not_detected_llw.drop_duplicates().reset_index(drop=True)
    
    df_max = get_strongest(df_detected_llw)
    
    curr_WAP_source = df_filter(df_max, s)['WAP'].unique()
    
    all_data = df_filter(df_agg_max, s)
    all_data = all_data[all_data.WAP.isin(curr_WAP_source)]
    all_data = all_data.reset_index(drop=True)

    detected = df_filter(df_detected_llw, s)
    detected = detected[(detected.WAP.isin(curr_WAP_source))]
    coverage = get_poll(detected)

    not_detected = df_filter(df_not_detected_llw, s)
    not_detected = not_detected[(not_detected.WAP.isin(curr_WAP_source))]
    hole = get_hole(not_detected, len(curr_WAP_source))
    coverage = coverage.append(hole)

    coverage = coverage.reset_index()
    
    #print('status', np.unique(coverage.STATUS, return_counts=True))
    #####-----PRINT JSON-----#####
    json_data = {'status': str(np.unique(coverage.STATUS, return_counts=True))} 
    json_data = json.dumps(json_data)
    with open(json_dir+'status.json', 'w') as fp:
        fp.write(json_data)
        
    if len(coverage.STATUS == 2) <= 1:
        pollute_happened = False
        print('pollute didn\'t happend.') 
        coverage = coverage[coverage['STATUS'] != 2]
    else:
        pollute_happened = True
    
    train_locs, val_locs = train_test_split(coverage, stratify=coverage.STATUS)
    
    #print('train locs', len(train_locs), 'val locs', len(val_locs))
    #####-----PRINT JSON-----#####
    json_data = {'train_locs': len(train_locs),
                 'val_locs': len(val_locs)} 
    json_data = json.dumps(json_data)
    with open(json_dir+'train_val_locs.json', 'w') as fp:
        fp.write(json_data)
        
    train_locs = train_locs.reset_index(drop=True)
    val_locs = val_locs.reset_index(drop=True)
    
    detected_train = detected.merge(train_locs, on=LL).drop_duplicates()
    detected_val = detected.merge(val_locs, on=LL).drop_duplicates()
    
    not_detected_train = not_detected.merge(train_locs, on=LL).drop_duplicates()
    not_detected_val = not_detected.merge(val_locs, on=LL).drop_duplicates()
    
    df_train = detected_train.append(not_detected_train)
    df_train = shuffle(df_train, random_state=r)
    
    df_val = detected_val.append(not_detected_val)
    df_val_generated = generate_polluted_val(df_val, curr_WAP_source)
    df_val = shuffle(df_val, random_state=r)
    
    df_train = generate(df_train, df_max)
    df_val = generate(df_val, df_max)
    df_val_generated = generate(df_val_generated, df_max)
    
    #print('df_train', np.unique(df_train.STATUS, return_counts=True), 
    #      'df_val', np.unique(df_val.STATUS, return_counts=True))
    #####-----PRINT JSON-----#####
    json_data = {'df_train': str(np.unique(df_train.STATUS, return_counts=True)), 
                  'df_val': str(np.unique(df_val.STATUS, return_counts=True))} 
    json_data = json.dumps(json_data)
    with open(json_dir+'df_train_val_locs.json', 'w') as fp:
        fp.write(json_data)
    
    return train_locs, val_locs, df_train, df_val, df_val_generated, pollute_happened

def generate_polluted_val(df_val_real, curr_WAP_source) :
    df_val_generated = pd.DataFrame()
    df_val_generated['LONGITUDE'] = df_val_real.LONGITUDE.repeat(len(curr_WAP_source))
    df_val_generated['LATITUDE'] = df_val_real.LATITUDE.repeat(len(curr_WAP_source))
    df_val_generated['WAP'] = np.tile(curr_WAP_source, len(df_val_real))
    return df_val_generated

def duplicate(df, n) :
    df_result = pd.DataFrame()
    for c in df.columns :
        df_result[c] = df[c].repeat(n)
    df_result = df_result.reset_index(drop=True)
    return df_result

def pairing(df_base, df_add) :
    n = len(df_base)
    df_base = duplicate(df_base, len(df_add))
    df_base['LONGITUDE_SOURCE'] = pd.np.tile(df_add.LONGITUDE, n) 
    df_base['LATITUDE_SOURCE'] = pd.np.tile(df_add.LATITUDE, n) 
    df_base['SIGNAL_SOURCE'] = pd.np.tile(df_add.SIGNAL, n) 
    return df_base

def merge_agg(df, group, value, aggregates, columns=None) :
    df_count = pd.DataFrame(df.groupby(group)[value].agg(aggregates)).reset_index()
    df_count.columns = group + aggregates if columns is None else group + columns
    df = df.merge(df_count, on=group, how="left").fillna(0)
    return df

def restructure(df) :
    df_final = pd.DataFrame()
    for i in range(1,521) :
        AP = 'WAP%03d' % i
        df_temp = df[[AP]+LLBF]
        df_temp = df_temp.rename(columns={AP : 'SIGNAL'})
        df_temp['WAP'] = i
        
        df_final = df_final.append(df_temp, ignore_index=True)
    df_final = df_final.drop_duplicates().reset_index(drop=True)
    return df_final

def get_strongest(df) :
    df_max = pd.DataFrame()
    for WAP in range(37, 43) :
        df_temp = df[df['WAP'] == WAP].reset_index(drop=True)
        max_val = df_temp['SIGNAL'].max()
        
        df_temp = df_temp[df_temp['SIGNAL'] == max_val]
        df_temp = df_temp.drop_duplicates().reset_index(drop=True)
        
        df_max = df_max.append(df_temp).reset_index(drop=True)

    return df_max
    
    
def generate(df, df_max):
    signal_exist = 'SIGNAL' in df.columns 
    curr_WAP_source = df_max['WAP'].unique()

    df_final = pd.DataFrame()
    for i in curr_WAP_source :
        df_temp = df[df['WAP']==i]

        #add pairing
        curr_df_max_by_WAP = df_max[df_max['WAP']==i]
        df_temp = pairing(df_temp, curr_df_max_by_WAP)
        
        if signal_exist:
            df_temp['DIFF_SIGNAL'] = df_temp['SIGNAL_SOURCE'] - df_temp['SIGNAL']

        df_final = df_final.append(df_temp, ignore_index=True)

    return df_final

def plot_data(train, val):
    fig = plt.figure(figsize=(25,5))
    plot_label(1, train.LONGITUDE, train.LATITUDE, train.STATUS, 'Train Data')
    plot_label(2, val.LONGITUDE, val.LATITUDE, val.STATUS, 'Val Data')
    plt.savefig(image_dir+'train_val.png')
    plt.close()
    #plt.show()

def plot_label(i, x, y, status, title) :
    positive = (status == 2) 
    normal = (status == 1) 
    negative = (status == 0)
    
    plt.subplot(1, 2, i)
    plt.scatter(x[positive], y[positive], c='b', s=35, label='polluted')
    plt.scatter(x[normal], y[normal], c='g', s=35, label='normal')
    plt.scatter(x[negative], y[negative], c='r', s=35, label='hole')
    
    plt.legend(loc=2, prop={'size': 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=20)

def other_result(name, f_regressor, 
                 f_clf_coverage, 
                 f_clf_coverage_2, 
                 f_clf_poll, f_clf_hole,  
                 df_train, df_val, 
                 df_train_per_locs, df_val_per_locs, pollution_flag, s) :
    dir = parentDir+"\\results\\model"
    #LLW
    x_train = df_train[LL+['WAP']].values
    x_val = df_val[LL+['WAP']].values

    #signal LLW
    f_regressor.fit(x_train, df_train['SIGNAL'])
    joblib.dump(f_regressor, os.path.join(dir, name+"_set" + str(s) + "_regressor_model.pth")) ###
    y_pred_signal = f_regressor.predict(x_val)    
    mse = mean_squared_error(df_val['SIGNAL'], y_pred_signal)

    #coverage LLW
    y_train = df_train['SIGNAL'] == -1
    y_val = df_val['SIGNAL'] == -1
    f_clf_coverage.fit(x_train, y_train)
    joblib.dump(f_clf_coverage, os.path.join(dir, name+"_set" + str(s) + "_clf_coverage_model.pth")) ###
    y_pred_coverage = f_clf_coverage.predict(x_val)
    c_acc = np.mean(y_val == y_pred_coverage)

    #LL
    x_train = df_train_per_locs[LL].values
    x_val = df_val_per_locs[LL].values

    #coverage LL
    y_train = df_train_per_locs['COVERAGE']
    y_val = df_val_per_locs['COVERAGE']
    
    f_clf_coverage_2.fit(x_train, y_train)
    joblib.dump(f_clf_coverage_2, os.path.join(dir, name+"_set" + str(s) + "_clf_coverage2_model.pth")) ###
    y_pred_coverage_ll = f_clf_coverage_2.predict(x_val)
    c_mse = mean_squared_error(y_val, y_pred_coverage_ll)
    
    # pollution LL  
    if pollution_flag:
        y_train = df_train_per_locs['STATUS'] == 2
        y_val = df_val_per_locs['STATUS'] == 2
    
        f_clf_poll.fit(x_train, y_train)
        joblib.dump(f_clf_poll, os.path.join(dir, name+"_set" + str(s) + "_clf_pollution.pth")) ###
        y_pred_poll = f_clf_poll.predict(x_val)
        p_acc = np.mean(y_pred_poll == y_val)

    # hole LL
    y_train = df_train_per_locs['STATUS'] == 0
    y_val = df_val_per_locs['STATUS'] == 0
    
    f_clf_hole.fit(x_train, y_train)
    joblib.dump(f_clf_hole, os.path.join(dir, name+"_set" + str(s) + "_clf_hole.pth")) ###
    y_pred_hole = f_clf_hole.predict(x_val)
    h_acc = np.mean(y_pred_hole == y_val)
    
    if pollution_flag:
        #print(name, 'mse signal LLW', mse, 'coverage LLW', c_acc) 
        #print('coverage LL', c_mse, 'acc polluted LL', p_acc, 'hole LL', h_acc)
        #####-----PRINT JSON-----#####
        json_data = {name: {'mse signal LLW': mse, 
                             'coverage LLW': c_acc,
                            'coverage LL': c_mse,
                             'acc polluted LL': p_acc,
                             'hole LL': h_acc}}
        json_data = json.dumps(json_data)
        with open(json_dir+name+'.json', 'w') as f:
            f.write(json_data)
        
        return y_pred_signal, y_pred_coverage, y_pred_coverage_ll, y_pred_poll, y_pred_hole
    else:
        #print(name, 'mse signal LLW', mse, 'coverage LLW', c_acc) 
        #print('coverage LL', c_mse, 'hole LL', h_acc)
        #####-----PRINT JSON-----#####
        json_data = {name: {'mse signal LLW': mse, 
                             'coverage LLW': c_acc,
                            'coverage LL': c_mse,
                             'hole LL': h_acc}}
        json_data = json.dumps(json_data)
        with open(json_dir+name+'.json', 'w') as f:
            f.write(json_data)
        return y_pred_signal, y_pred_coverage, y_pred_coverage_ll, y_pred_hole
    

def plot_encoder(i, preds, plot_type) :
    _, df_ll = preds
    x = df_ll['LONGITUDE']
    y = df_ll['LATITUDE']
    
    #Coverage LL
    if plot_type is 'coverage' :
        diff = df_ll['COVERAGE'] - df_ll['pred_COVERAGE']
        c_mse = mean_squared_error(df_ll['COVERAGE'], df_ll['pred_COVERAGE'])
        plot(i, x, y, diff, 'encoder diff coverage')
        #print('encoder mse', c_mse)
        #####-----PRINT JSON-----#####
    
    #Pollution LL
    elif plot_type is 'polluted' :
        diff = (df_ll['STATUS']==2).astype(int) - df_ll['pred_POLLUTED'].astype(int)
        p_acc = np.mean(((df_ll['STATUS']==2) == df_ll['pred_POLLUTED']))
        plot_bool(i, x, y, (df_ll['STATUS']==2).astype(int), df_ll['pred_POLLUTED'].astype(int), 'encoder polluted')
        #print('encoder acc', p_acc)
        #####-----PRINT JSON-----#####
    
    #Hole LL
    elif plot_type is 'hole' :
        diff = (df_ll['STATUS']==0).astype(int) - df_ll['pred_HOLE'].astype(int)
        h_acc = np.mean(((df_ll['STATUS']==0) == df_ll['pred_HOLE']))
        plot_bool(i, x, y, (df_ll['STATUS']==0).astype(int), df_ll['pred_HOLE'].astype(int), 'encoder hole')
        #print('encoder acc', h_acc)
        #####-----PRINT JSON-----#####
        
    else :
        print(plot_type, ' is not supported')

def plot_other(i, df_val_per_locs, preds, name, pollution_flag, plot_type) :
    if pollution_flag:
        _, _, pred_coverage_ll, pred_poll, pred_hole = preds
    else:
        _, _, pred_coverage_ll, pred_hole = preds
    x = df_val_per_locs['LONGITUDE']
    y = df_val_per_locs['LATITUDE']
    
    #Coverage LL
    if plot_type is 'coverage' :
        diff = df_val_per_locs['COVERAGE'] - pred_coverage_ll
        c_mse = mean_squared_error(df_val_per_locs['COVERAGE'], pred_coverage_ll)
        plot(i, x, y, diff, name + ' diff coverage')
        #print(name, 'mse', c_mse)
        #####-----PRINT JSON-----#####
    
    #Pollution LL
    elif plot_type is 'polluted' :
        p_acc = np.mean((df_val_per_locs['STATUS']==2) == pred_poll)
        plot_bool(i, x, y, (df_val_per_locs['STATUS']==2).astype(int), pred_poll, name + ' polluted')
        #print(name, 'acc', p_acc)
        #####-----PRINT JSON-----#####
    
    #Hole LL
    elif plot_type is 'hole' :
        h_acc = np.mean((df_val_per_locs['STATUS']==0) == pred_hole)
        plot_bool(i, x, y, (df_val_per_locs['STATUS']==0).astype(int), pred_hole, name + ' hole')
        #print(name, 'acc', h_acc)
        #####-----PRINT JSON-----#####
        
    else :
        print(plot_type, ' is not supported')
    
def plot(i, x, y, values, title) :
    cm = plt.cm.get_cmap('RdYlGn_r')
    cm = plt.cm.get_cmap('jet')
    plt.subplot(2, 2, i)
    sc = plt.scatter(x, y, c=values, s=50, cmap=cm)
    sc = plt.colorbar(sc)
    sc.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=20)
    
def plot_bool(i, x, y, values, preds, title) :
    true_positive = (values == 1) & (preds == 1) 
    false_positive = (values == 0) & (preds == 1) 
    true_negative = (values == 0) & (preds == 0)
    false_negative = (values == 1) & (preds == 0)
    
    plt.subplot(2, 2, i)
    plt.scatter(x[true_positive], y[true_positive], c='salmon', s=50, label='true positive')
    plt.scatter(x[false_positive], y[false_positive], c='r', s=50, label='false positive')
    plt.scatter(x[true_negative], y[true_negative], c='c', s=50, label='true negative')
    plt.scatter(x[false_negative], y[false_negative], c='b', s=50, label='false negative')
    
    plt.legend(loc=2, prop={'size': 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=20)


# In[ ]:




