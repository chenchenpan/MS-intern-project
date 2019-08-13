#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import preprocessing
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, log_loss, accuracy_score
import time
# from sklearn.manifold import TSNE
import joblib


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

print(pd.__version__)

def main():
    print('Starting to load encoded data...')
    Xtrain = np.load('data/Xtrain.npy')
    Xdev = np.load('data/Xdev.npy')
    Xtest = np.load('data/Xtest.npy')

    ytrain = np.load('data/ytrain.npy')
    ydev = np.load('data/ydev.npy')
    ytest = np.load('data/ytest.npy')

    print('Starting to tune the KNN model')

    knn_best_results, knn_saved_path = hyp_tuning('KNN', Xtrain, ytrain, Xdev, ydev)
    knn_best_model = joblib.load(knn_saved_path)
    pred_test = knn_best_model.predict(Xtest)
    mse_test = mean_squared_error(ytest, pred_test)
    knn_best_results['Test_mse'] = mse_test
    print('KNN model best results are {}'.format(knn_best_results))

    print('Starting to tune the RF model')
    rf_best_results, rf_saved_path = hyp_tuning('RF', Xtrain, ytrain, Xdev, ydev)
    rf_best_model = joblib.load(rf_saved_path)
    pred_test = rf_best_model.predict(Xtest)
    mse_test = mean_squared_error(ytest, pred_test)
    rf_best_results['Test_mse'] =  pred_test
    print('RF model best results are {}'.format(rf_best_results))

    print('Starting to run Linear Regression model')
    regr_multilr = MultiOutputRegressor(LinearRegression())
    regr_multilr.fit(Xtrain, ytrain)
    pred_train = regr_multilr.predict(Xtrain)
    mse_train = mean_squared_error(ytrain, pred_train)

    pred_dev = regr_multilr.predict(Xdev)
    mse_dev = mean_squared_error(ydev, pred_dev)

    pred_test = regr_multilr.predict(Xtest)
    mse_test = mean_squared_error(ytest, pred_test)

    lr_results = {}
    lr_results['Dev_mse'] = mse_dev
    lr_results['Train_mse'] = mse_train
    lr_results['Test_mse'] = mse_test

    joblib.dump(regr_multilr, 'results/baseline_LR.joblib')
    print('Linear Regression results are {}'.format(lr_results))



def hyp_tuning(model_name, Xtrain, ytrain, Xdev, ydev):
    best_results = {}
    best_dev_mse = 1000
    
    for hyp in range(1, 10):
        
        print('='*20)
        print('starting to compute hyp={}'.format(hyp))
        
        if model_name == 'KNN':
            model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=hyp))
        elif model_name == 'RF':
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=hyp,
                                                          #max_depth=max_depth,
                                                          random_state=0))
        else:
            raise ValueError('Unknown model_name')                               

        model.fit(Xtrain, ytrain)
        
        pred_train = model.predict(Xtrain)
        mse_train = mean_squared_error(ytrain, pred_train)

        pred_dev = model.predict(Xdev)
        mse_dev = mean_squared_error(ydev, pred_dev)
        
        if mse_dev < best_dev_mse:
            best_dev_mse = mse_dev
            print('Up to now the best dev_mse is {}'.format(best_dev_mse))
            print('and the associated training_mse is {}'.format(mse_train))
            best_results['Dev_mse'] = best_dev_mse
            best_results['Train_mse'] = mse_train
            
            best_model = model
            best_hyp = hyp
            
    save_path = 'results/baseline_{}_with_{}.joblib'.format(model_name, best_hyp)
    joblib.dump(best_model, save_path)
    
    return best_results, save_path


if __name__ == '__main__':
    main()

