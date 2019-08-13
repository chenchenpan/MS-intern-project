#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import confusion_matrix, mean_squared_error, log_loss, accuracy_score


## check the package version
print(pd.__version__)

# # Preprocess data
# 
# 1. split dataset
# 2. transfer datetime data
# 3. encode categorical data
# 4. encode boolean type data
# 5. normalize data

def main():

    df = pd.read_csv('TenantInfo-and-usage_shuffled_inf.csv'#, nrows=100
    )
    print('the dataset size is {}'.format(df.shape))

    Xtrain, ytrain, Xdev, ydev, Xtest, ytest = split_dataset(df)
    print('Training size is {}'.format(Xtrain.shape))
    print('Dev size is {}'.format(Xdev.shape))
    print('Test size is {}'.format(Xtest.shape))
    print('The output size is {}'.format(ytest.shape[1]))

    ytrain = ytrain.to_numpy()
    ydev = ydev.to_numpy()
    ytest = ytest.to_numpy()
    print('Outputs are ready!')

    np.save('data/ytrain.npy', ytrain)
    np.save('data/ydev.npy', ydev)
    np.save('data/ytest.npy', ytest)

    print('Saved the outputs targets!')

    Xtrain_arr, dv, vocab = encoder_training_inputs(Xtrain)
    Xdev_arr = encoder_dev_test_inputs(Xdev, dv, 'dev')
    Xtest_arr = encoder_dev_test_inputs(Xtest, dv, 'test')

    print('After encoding, the training size is {}'.format(Xtrain_arr.shape))
    print('After encoding, the dev size is {}'.format(Xdev_arr.shape))
    print('After encoding, the test size is {}'.format(Xtest_arr.shape))

    scaler = StandardScaler()
    Xtrain_scal = scaler.fit_transform(Xtrain_arr)
    Xdev_scal = scaler.transform(Xdev_arr)
    Xtest_scal = scaler.transform(Xtest_arr)

    np.save('data/Xtrain.npy', Xtrain_scal)
    np.save('data/Xdev.npy', Xdev_scal)
    np.save('data/Xtest.npy', Xtest_scal)
    print('Saved the encoded inputs!')



def split_dataset(df, age=360):
    cols_name = pd.Series(data=df.columns)

    ar_04_beg_col_index = cols_name[cols_name == 'AR_exchange_04'].index[0]
    ar_06_beg_col_index = cols_name[cols_name == 'AR_exchange_06'].index[0]
    ar_06_end_col_index = cols_name[cols_name == 'AR_officeclient_06'].index[0]

    wl_AR_cols = cols_name[ar_04_beg_col_index : ar_06_end_col_index+1].tolist()

    output_cols = cols_name[ar_06_beg_col_index : ar_06_end_col_index+1].tolist()

    # # Use mature tenants as training and dev, use the new tenants as test.
    df_train = df.loc[df['Age'] >= age]
    df_test = df.loc[df['Age'] < age]

    print(df_train.shape)
    print(df_test.shape)

    ytrain = df_train.loc[:, output_cols]
    ytest = df_test.loc[:, output_cols]

    Xtrain = df_train.drop(columns=wl_AR_cols) # use profile only
    # Xtrain = df_train.drop(columns=output_cols) # use profile + previous usage
    # Xtrain = df_train.loc[:, wl_AR_cols[:-12]] # use previous usage only

    Xtest = df_test.drop(columns=wl_AR_cols) # use profile only
    # Xtest = df_test.drop(columns=output_cols) # use profile + previous usage
    # Xtest = df_test.loc[:, wl_AR_cols[:-12]] # use previous usage only

    dev_size = int(Xtrain.shape[0] * 0.2)

    Xdev = Xtrain.iloc[-dev_size:,:]
    ydev = ytrain.iloc[-dev_size:,:]

    Xtrain = Xtrain.iloc[:-dev_size,:]
    ytrain = ytrain.iloc[:-dev_size,:]

    return Xtrain, ytrain, Xdev, ydev, Xtest, ytest


def process_object_cols(df):
    cols_datetime = ['CreatedDate', 'CreateDateOfFirstSubscription','FirstPaidEXOStartDate',
       'FirstPaidSPOStartDate', 'FirstPaidOD4BStartDate',
       'FirstPaidSfBStartDate', #'FirstPaidYammerStartDate',
       'FirstPaidTeamsStartDate', 'FirstPaidProPlusStartDate',
       #'FirstPaidAADPStartDate', 'FirstPaidAIPStartDate',
       #'FirstPaidAATPStartDate', 'FirstPaidIntuneStartDate',
       #'FirstPaidMCASStartDate', 'FirstPaidO365E5SkuStartDate',
       #'FirstPaidM365E5SkuStartDate', 'FirstPaidEMSE5SkuStartDate'
                    ]
    df_datetime = df.loc[:, cols_datetime]
    
    cols_cat = ['CountryCode', 'Languange', #'DataCenterInstance', 'DataCenterModel',
       'SignupLocationInfo_Country', #'SignupLocationInfo_CountryCode',
       #'SignupLocationInfo_Region', 'TopParents_AreaName',
       'TopParents_CountryCode', #'TopParents_BigAreaName', 
       'TopParents_Industry', #'TopParents_RegionName',
       'TopParents_SegmentGroup', #'TopParents_SubRegionName',
       'TopParents_VerticalName']
    df_cat = df.loc[:, cols_cat]
    
    df_tenantid = df.loc[:,'TenantId']
    
    return df_tenantid, df_cat, df_datetime



def encoder_datetime(df):
    cols = df.columns
    for i in cols:
        df[i] = pd.to_datetime(df[i], utc=True, errors='coerce').astype(int,errors='ignore')
    return df


def encoder_num_bool(df):
    X_num = df.select_dtypes(include=['float','int'])
    X_bool = df.select_dtypes(include='bool')
    
    X_bool = X_bool.astype(int).to_numpy()
    X_num = X_num.to_numpy()
    return X_bool, X_num


def concat_inputs(X_cat_encoded, X_num, X_bool, Xdatetime):
    X = np.concatenate((X_cat_encoded, X_num, X_bool, Xdatetime), axis=1)
    return X


def encoder_training_inputs(df_X):
    print('Starting to encode training inputs:')
    X_bool, X_num = encoder_num_bool(df_X)
    df_X_id, df_X_cat, df_X_datetime = process_object_cols(df_X)
    
    id_file_name = df_X_id.name + '_train.csv'
    df_X_id.to_csv(id_file_name, header=False)
    
    X_datetime = encoder_datetime(df_X_datetime)
    X_datetime = X_datetime.to_numpy()
    
    X_cat_dict = df_X_cat.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_cat_encoded = dv.fit_transform(X_cat_dict)
    vocab = dv.vocabulary_
    
    X_arr = np.concatenate((X_cat_encoded, X_num, X_bool, X_datetime), axis=1)
    
    return X_arr, dv, vocab
  


def encoder_dev_test_inputs(df_X, dv, dataset_type):
    # dataset_type is a string, it should be 'dev', 'test' etc.
    
    print('Starting to encode dev or test inputs:')
    X_bool, X_num = encoder_num_bool(df_X)
    df_X_id, df_X_cat, df_X_datetime = process_object_cols(df_X)
    
    id_file_name = df_X_id.name + '_' + dataset_type + '.csv'
    df_X_id.to_csv(id_file_name, header=False)
    
    X_datetime = encoder_datetime(df_X_datetime)
    X_datetime = X_datetime.to_numpy()
    
    X_cat_dict = df_X_cat.to_dict(orient='records')
    X_cat_encoded = dv.transform(X_cat_dict)
    
    X_arr = np.concatenate((X_cat_encoded, X_num, X_bool, X_datetime), axis=1)
    
    return X_arr
    

if __name__ == '__main__':
    main()