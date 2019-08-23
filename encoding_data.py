#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import joblib
import collections
import argparse
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer


## check the package version
print(pd.__version__)

# # Preprocess data steps:

# 1. split dataset
# 2. transfer datetime data
# 3. encode categorical data
# 4. encode boolean type data
# 5. normalize data

def main():

    parser = argparse.ArgumentParser()

    # parameters for select input features
    parser.add_argument('--predict_ahead', type=int,
        default=0,
        help=('predict at the beginbing of the month (the number of '
            'active users in that month cannot be included in inputs) (0 or 1)'))


    parser.add_argument('--previous_usage', type=int,
        default=0,
        help=('use the previous two month usage as inputs (0 or 1)'))

    # parameter for saving the encoded data
    parser.add_argument('--output_dir', type=str,
        default='/data/home/t-chepan/projects/MS-intern-project/data',
        help=('directory to store the encoded data.'))

    args = parser.parse_args()

    predict_ahead = args.predict_ahead
    previous_usage = args.previous_usage


    df = pd.read_csv('TenantInfo-and-usage_shuffled_inf.csv'#, nrows=100
    )
    print('the full dataset size is {}'.format(df.shape))

    df_train, df_dev, df_test = split_dataset(df)

    ### save the dev set raw data for demo purpose. ### 
    save_path = os.path.join(args.output_dir, 'dev_set_raw_data.csv')
    df_dev.to_csv(save_path, index=False)

    df_train_X, df_train_y = separate_input_output_cols(df_train, predict_ahead, previous_usage)
    df_dev_X, df_dev_y = separate_input_output_cols(df_dev, predict_ahead, previous_usage)
    df_test_X, df_test_y = separate_input_output_cols(df_test, predict_ahead, previous_usage)

    ytrain = df_train_y.to_numpy()
    ydev = df_dev_y.to_numpy()
    ytest = df_test_y.to_numpy()
    print('Outputs are ready!')

    ytrain_path = os.path.join(args.output_dir, 'ytrain.npy')
    np.save(ytrain_path, ytrain)
    ydev_path = os.path.join(args.output_dir, 'ydev.npy')
    np.save(ydev_path, ydev)
    ytest_path = os.path.join(args.output_dir, 'ytest.npy')
    np.save(ytest_path, ytest)

    print('Saved the outputs targets!')

    Xtrain, dv, scaler, cols_name, df_train_id = encode_training_inputs(df_train_X)
    Xdev, df_dev_id = encode_dev_test_inputs(df_dev_X, dv, scaler)
    Xtest, df_test_id = encode_dev_test_inputs(df_test_X, dv, scaler)

    print('After encoding, the training size is {}'.format(Xtrain.shape))
    print('After encoding, the dev size is {}'.format(Xdev.shape))
    print('After encoding, the test size is {}'.format(Xtest.shape))

    train_id_path = os.path.join(args.output_dir, 'train_id.csv')
    df_train_id.to_csv(train_id_path, header=False)
    dev_id_path = os.path.join(args.output_dir, 'dev_id.csv')
    df_dev_id.to_csv(dev_id_path, header=False)
    test_id_path = os.path.join(args.output_dir, 'test_id.csv')
    df_test_id.to_csv(test_id_path, header=False)

    col_name_path = os.path.join(args.output_dir, 'encoded_columns_name.txt')
    with open(col_name_path, 'w') as f:
        for item in cols_name:
            f.write("%s\n" % item)
       
    dv_path = os.path.join(args.output_dir, 'vectorizer.pkl')
    joblib.dump(dv, dv_path)

    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    Xtrain_path = os.path.join(args.output_dir, 'Xtrain.npy')
    np.save(Xtrain_path, Xtrain)
    Xdev_path = os.path.join(args.output_dir, 'Xdev.npy')
    np.save(Xdev_path, Xdev)
    Xtest_path = os.path.join(args.output_dir, 'Xtest.npy')
    np.save(Xtest_path, Xtest)
    print('Saved the encoded inputs!')


def split_dataset(df, age=360):
    df_train = df.loc[df['Age'] >= age]
    df_test = df.loc[df['Age'] < age]
    
    df_train = df_train.drop('Age', axis=1)
    df_test = df_test.drop('Age', axis=1)
    
    print('df_test shape is {}'.format(df_test.shape))
    
    ### split dev set from training set ###
    dev_size = int(df_train.shape[0] * 0.01)

    df_dev = df_train.iloc[-dev_size:,:]
    df_train = df_train.iloc[:-dev_size,:]
    print('df_dev shape is {}'.format(df_dev.shape))
    print('df_train shape is {}'.format(df_train.shape))
    
    return df_train, df_dev, df_test


def separate_input_output_cols(df, predict_ahead, previous_usage):
    cols_name = pd.Series(data=df.columns)

    ar_04_beg_col_index = cols_name[cols_name == 'AR_exchange_04'].index[0]
    ar_06_beg_col_index = cols_name[cols_name == 'AR_exchange_06'].index[0]
    ar_06_end_col_index = cols_name[cols_name == 'AR_officeclient_06'].index[0]
    
    au_06_beg_col_index = cols_name[cols_name == 'AU_exchange'].index[0]
    au_06_end_col_index = cols_name[cols_name == 'AU_officeclient'].index[0]
    
    au_06_cols = cols_name[au_06_beg_col_index : au_06_end_col_index+1].tolist()

    wl_AR_cols = cols_name[ar_04_beg_col_index:ar_06_end_col_index+1].tolist()

    output_cols = cols_name[ar_06_beg_col_index:ar_06_end_col_index+1].tolist()

    df_y = df.loc[:, output_cols]
    print('outputs shape is {}'.format(df_y.shape))

    if previous_usage == 1:
    ### use profile info and the usage of previous 2 months ###
        df_X = df.drop(columns=output_cols)
    else:
    ### use profile info only, exclude the usage of previous months ###
        df_X = df.drop(columns=wl_AR_cols) 
    
    if predict_ahead == 1:
    ### if we predict the usage at the beginning of the month, we should ###
    ### exclude the active users' number for that month (June).          ###
        df_X = df_X.drop(columns=au_06_cols) 
    
    print('inputs shape is {}'.format(df_X.shape))

    return df_X, df_y



def process_object_cols_for_inputs(df):
    ### only selected the StartDate for outputs workloads ###
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
    
    ### drop some redundant columns ###
    cols_cat = ['CountryCode', 'Languange', #'DataCenterInstance', 'DataCenterModel',
       #'SignupLocationInfo_Country', #'SignupLocationInfo_CountryCode',
       #'SignupLocationInfo_Region', 'TopParents_AreaName',
       #'TopParents_CountryCode', #'TopParents_BigAreaName', 
       'TopParents_Industry', #'TopParents_RegionName',
       #'TopParents_SegmentGroup', #'TopParents_SubRegionName',
       #'TopParents_VerticalName'
               ]
    df_cat = df.loc[:, cols_cat]
    
    df_tenantid = df.loc[:,'TenantId']
    
    return df_tenantid, df_cat, df_datetime



def encode_datetime(df_datetime):
    """Encode the datetime inputs from '2/5/2014 5:31:19 AM' format
        to a numerical number of UTC format.
        
    Args:
      df_datetime: a DataFrame that only stores the datetime inputs.
        
    Returns:
      X_datetime: a numpy array that contains the encoded datetime inputs.
      datetime_cols: a list that contains the datetime colunms name.
      
    """
    
    cols = df_datetime.columns
    for i in cols:
        df_datetime[i] = pd.to_datetime(df_datetime[i], utc=True,
                                        errors='coerce').astype(int,errors='ignore')
        
    X_datetime = df_datetime.to_numpy()
    datetime_cols = cols.to_list()
    
    return X_datetime, datetime_cols



def encode_num_bool(df):
    """Encode the numerical and boolean inputs to an array.
        
    Args:
      df: a DataFrame that stores the inputs raw data.
        
    Returns:
      X_bool: a numpy array that contains the encoded boolean inputs.
      X_num: a numpy array that contains the encoded numerical inputs.
      num_cols: a list that contains the numerical columns name.
      bool_cols: a list that contains the boolean columns name.
    """
    
    X_num = df.select_dtypes(include=['float','int'])
    X_bool = df.select_dtypes(include='bool')

    num_cols = X_num.columns.to_list()
    bool_cols = X_bool.columns.to_list()
    
    X_bool = X_bool.astype(int).to_numpy()
    X_num = X_num.to_numpy()
    return X_bool, X_num, num_cols, bool_cols



def encode_training_inputs(df_X):
    """Encode the inputs for training set.
        
    Args:
      df_X: a DataFrame that stores the inputs raw data.
        
    Returns:
      X_scal: a numpy array that contains the encoded and normalized inputs.
      dv: a DictVectorizer that is trained on the training set.
      scaler: a StandardScaler that is trained on the training set.
      cols_name: a list that contains all of the inputs features after encoding.
      df_X_id: a DataFrame that contains the TenantId for training set.
    """
    
    
    print('Starting to encode training inputs...')
    X_bool, X_num, num_cols, bool_cols = encode_num_bool(df_X)
    
    df_X_id, df_X_cat, df_X_datetime = process_object_cols_for_inputs(df_X)
    
    X_datetime, datetime_cols = encode_datetime(df_X_datetime)
    
    
    ### encode the categorical columns ###
    X_cat_dict = df_X_cat.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_cat_encoded = dv.fit_transform(X_cat_dict)
    vocab = dv.vocabulary_
    vocab_od = collections.OrderedDict(sorted(vocab.items(), key=lambda x:x[1]))
    cat_encoded_cols = list(vocab_od.keys())
    
    ### normalize all the inputs ###
    X_arr = np.concatenate((X_cat_encoded, X_num, X_bool, X_datetime), axis=1)
    scaler = StandardScaler()
    X_scal = scaler.fit_transform(X_arr)

    cols_name = cat_encoded_cols + num_cols + bool_cols + datetime_cols

    assert len(cols_name) == X_scal.shape[1]
    
    return X_scal, dv, scaler, cols_name, df_X_id



def encode_dev_test_inputs(df_X, dv, scaler):
    """Encode the inputs for dev or test set.
        
    Args:
      df_X: a DataFrame that stores the inputs raw data.
      dv: a DictVectorizer that is trained on the training set.
      scaler: a StandardScaler that is trained on the training set.
        
    Returns:
      X_scal: a numpy array that contains the encoded and normalized inputs.
      df_X_id: a DataFrame that contains the TenantId for dev or test set.
    """
    
    print('Starting to encode dev or test inputs...')
    X_bool, X_num, _, _ = encode_num_bool(df_X)
    df_X_id, df_X_cat, df_X_datetime = process_object_cols_for_inputs(df_X)
    
    X_datetime, _ = encode_datetime(df_X_datetime)
    
    ### encode the categorical columns ###
    X_cat_dict = df_X_cat.to_dict(orient='records')
    X_cat_encoded = dv.transform(X_cat_dict)

    ### normalize all the inputs ###
    X_arr = np.concatenate((X_cat_encoded, X_num, X_bool, X_datetime), axis=1)
    
    X_scal = scaler.transform(X_arr)
    
    return X_scal, df_X_id


if __name__ == '__main__':
    main()