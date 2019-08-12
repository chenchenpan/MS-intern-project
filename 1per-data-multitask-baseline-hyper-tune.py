#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, log_loss, accuracy_score
import time
from sklearn.manifold import TSNE
from sklearn.externals import joblib


## check the env
pd.__version__
#%%

def main():
    
    ## load data
    df = pd.read_csv('encoded_data.csv', nrows=20000)
    print(df.shape)


    # ## Preparing data
    cols_name = pd.Series(data=df.columns)
    ar_04_beg_col_index = cols_name[cols_name == 'AR_exchange_04'].index[0]
    # ar_06_beg_col_index = cols_name[cols_name == 'AR_exchange_06'].index[0]
    ar_06_end_col_index = cols_name[cols_name == 'AR_eslt_06'].index[0]

    wl_AR_cols = cols_name[ar_04_beg_col_index : ar_06_end_col_index+1].tolist()

    output_cols = wl_AR_cols[-12:]


    ## Seperate Training set and test set
    df_train = df.loc[df['Train'] == 1]
    df_test = df.loc[df['Train'] == 0]

    df_train.drop(columns='Train', inplace=True)
    df_test.drop(columns='Train', inplace=True)


    ## Define the outputs
    ytrain = df_train.loc[:, output_cols]
    ytest = df_test.loc[:, output_cols]

    ## Define the inputs
    # Xtrain = df_train.drop(columns=output_cols) # use profile + previous usage
    Xtrain = df_train.drop(columns=wl_AR_cols) # use profile only
    # Xtrain = df_train.loc[:, wl_AR_cols[:-12]] # use previous usage only

    # Xtest = df_test.drop(columns=output_cols)
    Xtest = df_test.drop(columns=wl_AR_cols)
    # Xtest = df_test.loc[:, wl_AR_cols[:-12]] 

    ## Seperate the dev set from training set
    dev_size = int(Xtrain.shape[0] * 0.2)

    Xtrain = Xtrain.to_numpy()
    ytrain = ytrain.to_numpy()

    Xdev = Xtrain[-dev_size:]
    ydev = ytrain[-dev_size:]

    Xtrain = Xtrain[:-dev_size]
    ytrain = ytrain[:-dev_size]

    Xtest = Xtest.to_numpy()
    ytest = ytest.to_numpy()

    ## normalize the data
    scaler = StandardScaler()

    Xtrain = scaler.fit_transform(Xtrain)
    Xdev = scaler.transform(Xdev)
    Xtest = scaler.transform(Xtest)
    
    
    # ## Baseline models
    
    ## Linear Regression Model
    regr_multilr = MultiOutputRegressor(LinearRegression())
    regr_multilr.fit(Xtrain, ytrain)
    pred_train_multilr = regr_multilr.predict(Xtrain)
    pred_dev_multilr = regr_multilr.predict(Xdev)

    print('LR training MSE is {}'.format(mean_squared_error(ytrain, pred_train_multilr)))
    print('LR dev MSE is {}'.format(mean_squared_error(ydev, pred_dev_multilr)))

    save_path = 'results/baseline_LR'
    joblib.dump(regr_multilr, save_path)

    ## tuning the model and then use the best model on the test
    MODEL_NAME = 'KNN'
    knn_best = hyp_tuning(MODEL_NAME, Xtrain, ytrain, Xdev, ydev)
    print('KNN training MSE is {}'.format(knn_best['mse_train']))
    print('KNN dev MSE is {}'.format(knn_best['mse_dev']))

    pred_knn_test = predict_on_test(knn_best, MODEL_NAME)
    mse_knn_test = mean_squared_error(ytest, pred_knn_test)
    print('KNN test MSE is {}'.format(mse_knn_test))
    print('*' * 20)

    MODEL_NAME = 'RF'
    rf_best = hyp_tuning(MODEL_NAME, Xtrain, ytrain, Xdev, ydev)
    print('RF training MSE is {}'.format(rf_best['mse_train']))
    print('RFD dev MSE is {}'.format(rf_best['mse_dev']))

    pred_rf_test = predict_on_test(rf_best, MODEL_NAME)
    mse_rf_test = mean_squared_error(ytest, pred_rf_test)
    print('KNN test MSE is {}'.format(mse_rf_test))

    

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
            print(best_dev_mse)
            best_results[hyp] = {'mse_train': mse_train, 'mse_dev': mse_dev}
            save_path = 'results/baseline_{}_with_{}'.format(model_name, hyp)
            joblib.dump(model, save_path)
    
    return best_results



def predict_on_test(best_results, model_name):
    best_hyp = list(best_results.keys())[-1]
    save_path = 'results/baseline_{}_with_{}'.format(model_name, best_hyp)
    best_model = joblib.load(save_path)
    pred_test = best_model.predict(Xtest)
    return mean_squared_error(ytest, pred_test)

if __name__== '__main__':
    main()



#%%
