#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, log_loss, accuracy_score
import time
import joblib
import argparse
import json

print(pd.__version__)

def main():
    t1 = time.time()

    parser = argparse.ArgumentParser()

    # data and encode file

    # pick up the size for training
    parser.add_argument('--training_size', type=float,
        default=1.0,
        help=('how much percentage of training data will be used'))

    args = parser.parse_args()


    print('Starting to load encoded data...')
    Xtrain = np.load('data/Xtrain.npy')
    full_size = Xtrain.shape[0]
    selected_size = int(full_size * args.training_size)
    Xtrain = Xtrain[:selected_size,:]

    ytrain = np.load('data/ytrain.npy')
    ytrain = ytrain[:selected_size,:]

    Xdev = np.load('data/Xdev.npy')
    # Xtest = np.load('data/Xtest.npy')

    ydev = np.load('data/ydev.npy')
    # ytest = np.load('data/ytest.npy')


    print('Starting to tune the RF model')
    rf_best_results = hyp_tuning('RF', Xtrain, ytrain, Xdev, ydev, args.training_size)
    
    # pred_test = rf_best_model.predict(Xtest)
    # mse_test = mean_squared_error(ytest, pred_test)
    # rf_best_results['Test_mse'] =  mse_test
    print('RF model best results are {}'.format(rf_best_results))


    print('Starting to tune the KNN model')
    knn_best_results = hyp_tuning('KNN', Xtrain, ytrain, Xdev, ydev, args.training_size)
    
    # pred_test = knn_best_model.predict(Xtest)
    # mse_test = mean_squared_error(ytest, pred_test)
    # knn_best_results['Test_mse'] = mse_test
    print('KNN model best results are {}'.format(knn_best_results))

    
    print('Starting to run Linear Regression model')
    regr_multilr = MultiOutputRegressor(LinearRegression())
    regr_multilr.fit(Xtrain, ytrain)
    # pred_train = regr_multilr.predict(Xtrain)
    # mse_train = mean_squared_error(ytrain, pred_train)

    pred_dev = regr_multilr.predict(Xdev)
    mse_dev = mean_squared_error(ydev, pred_dev)

    # pred_test = regr_multilr.predict(Xtest)
    # mse_test = mean_squared_error(ytest, pred_test)

    lr_results = {}
    lr_results['Dev_mse'] = mse_dev
    # lr_results['Train_mse'] = mse_train
    # lr_results['Test_mse'] = mse_test

    joblib.dump(regr_multilr, 'results/baselinemodel_LR_with_{}data.joblib'.format(args.training_size))
    print('Linear Regression results are {}'.format(lr_results))

    ### save best results from diff models
    baseline_results = []
    baseline_results.append(rf_best_results)
    baseline_results.append(knn_best_results)
    baseline_results.append(lr_results)

    results_df = pd.DataFrame(baseline_results, index=['rf','knn','rl'])
    results_df.to_csv('results/baselineresults_with_{}data.csv'.format(args.training_size))

    t2 = time.time()

    print('Total used the time {} seconds'.format(t2 -t1))


def hyp_tuning(model_name, Xtrain, ytrain, Xdev, ydev, datasize):
    best_results = {}
    best_dev_mse = 1000
    
    for hyp in range(3, 15):
        
        print('='*20)
        print('starting to compute hyp={}'.format(hyp))
        
        if model_name == 'KNN':
            model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=hyp))
            
        elif model_name == 'RF':
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=hyp,
                                                        #   max_depth=200,
                                                          random_state=0))
        else:
            raise ValueError('Unknown model_name')                               

        model.fit(Xtrain, ytrain)
        
        # pred_train = model.predict(Xtrain)
        # mse_train = mean_squared_error(ytrain, pred_train)

        pred_dev = model.predict(Xdev)
        mse_dev = mean_squared_error(ydev, pred_dev)
        
        if mse_dev < best_dev_mse:
            best_dev_mse = mse_dev
            print('Up to now the best dev_mse is {}'.format(best_dev_mse))
            # print('and the associated training_mse is {}'.format(mse_train))
            best_results['Dev_mse'] = best_dev_mse
            # best_results['Train_mse'] = mse_train
            
            best_model = model
            best_hyp = hyp

    save_path = 'results/baselinemodel_{}_with_{}_with_{}data.joblib'.format(model_name, best_hyp, datasize)
    joblib.dump(best_model, save_path)


    
    return best_results


if __name__ == '__main__':
    main()

