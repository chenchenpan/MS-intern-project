#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import joblib
import collections
import argparse
import os
import json
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


## check the package version
# print(pd.__version__)

# # Preprocess data steps:

# 1. split dataset
# 2. transfer datetime data
# 3. encode categorical data
# 4. encode boolean type data
# 5. normalize data

def main():

    parser = argparse.ArgumentParser()

    # parameters for select input data and metedata configure files
    parser.add_argument('--input_dir', type=str,
        default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to load the raw data.'))

    parser.add_argument('--data_file', type=str,
        # default='TenantInfo-and-usage_shuffled_inf.csv',
        help=('which data file will be used? (with text features Or not?)'))

    parser.add_argument('--configure_file', type=str,
        # default='configure.json',
        help=('which configure file (metadata) will be used?'))

    # parameter for saving the encoded data
    parser.add_argument('--output_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/encoded_data',
        help=('directory to store the encoded data.'))

    # parameter for using text features
    parser.add_argument('--use_text_features', type=str2bool, nargs='?',
        const=True, default=False,
        help=('whether encode the text features or not?'))


    args = parser.parse_args()


    ### load raw data and related metadata configure file
    load_path = os.path.join(args.input_dir, args.data_file)
    df = pd.read_csv(load_path#, nrows=1000
    )
    print('the full dataset size is {}'.format(df.shape))

    # print('int columns are {}'.format(df.select_dtypes(include=['int']).columns.to_list()))
    # print('object columns are {}'.format(df.select_dtypes(include=['object']).columns.to_list()))
    # print('*' * 50)
    # print(df.info())
    # print('*' * 50)

    metadata_path = os.path.join(args.input_dir, 'configure.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)


    df_train, df_dev, df_test, df_young = split_dataset(df)

    ### save the dev set raw data for demo purpose.
    save_path = os.path.join(args.output_dir, 'dev_set_raw_data.csv')
    df_dev.to_csv(save_path, index=False)

    ytrain, Xtrain, dv, scaler, text_token, cols_name = encode_dataset(df_train, metadata)
    ydev, Xdev, _, _, _, _ = encode_dataset(df_dev, metadata, dv=dv, scaler=scaler, text_token=text_token)
    ytest, Xtest, _, _, _, _ = encode_dataset(df_test, metadata, dv=dv, scaler=scaler, text_token=text_token)
    yyoung, Xyoung, _, _, _, _ = encode_dataset(df_young, metadata, dv=dv, scaler=scaler, text_token=text_token)


    ### save the results.
    ytrain_path = os.path.join(args.output_dir, 'ytrain.npy')
    np.save(ytrain_path, ytrain)
    ydev_path = os.path.join(args.output_dir, 'ydev.npy')
    np.save(ydev_path, ydev)
    ytest_path = os.path.join(args.output_dir, 'ytest.npy')
    np.save(ytest_path, ytest)
    yyoung_path = os.path.join(args.output_dir, 'yyoung.npy')
    np.save(yyoung_path, yyoung)
    print('Saved the encoded outputs!')

    Xtrain_path = os.path.join(args.output_dir, 'Xtrain.npy')
    np.save(Xtrain_path, Xtrain)
    Xdev_path = os.path.join(args.output_dir, 'Xdev.npy')
    np.save(Xdev_path, Xdev)
    Xtest_path = os.path.join(args.output_dir, 'Xtest.npy')
    np.save(Xtest_path, Xtest)
    Xyoung_path = os.path.join(args.output_dir, 'Xyoung.npy')
    np.save(Xyoung_path, Xyoung)
    print('Saved the encoded inputs!')

    col_name_path = os.path.join(args.output_dir, 'encoded_columns_name.txt')
    with open(col_name_path, 'w') as f:
        for item in cols_name:
            f.write("%s\n" % item)
    
    ### save all the token 
    dv_path = os.path.join(args.output_dir, 'cat_vectorizer.pkl')
    joblib.dump(dv, dv_path)

    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    if args.use_text_features:
        print('=' * 20)
        print(args.use_text_features)
        print('=' * 20)
        for i in range(len(text_token)):
            token_path = os.path.join(args.output_dir, 'text_{}_vectorizer.pkl'.format(i))
            joblib.dump(text_token[i], token_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
   

def split_dataset(df, age=360, split_per=0.01):
    df_train = df.loc[df['Age'] >= age]
    df_young = df.loc[df['Age'] < age]
    
    df_train = df_train.drop('Age', axis=1)
    df_young = df_young.drop('Age', axis=1)
    
    print('df_young shape is {}'.format(df_young.shape))
    
    ### split dev set from training set
    split_size = int(df_train.shape[0] * split_per)

    df_dev = df_train.iloc[-split_size:,:]
    df_train = df_train.iloc[:-split_size,:]
    print('df_dev shape is {}'.format(df_dev.shape))

    ### split test set from training set
    df_test = df_train.iloc[-split_size:,:]
    df_train = df_train.iloc[:-split_size,:]

    print('df_test shape is {}'.format(df_test.shape))
    print('df_train shape is {}'.format(df_train.shape))
    return df_train, df_dev, df_test, df_young


def separate_input_output_cols(df, metadata):
    """According to the metadata, separate the input features, output features and 
        different types of input features.

    Args:
      df: a DataFrame that stores the raw data.
      metadata: a dictionary that stores the detail description for features.
        metadata = {'input_features': ['TenantId','CreatedDate', ...]
                    'output_label': ['AR_exchange_06','AR_sharepoint_06', ...]
                    'input_bool': ['HasEXO','HasSPO', ...],
                    'input_categorical': ['CountryCode', 'Languange', ...],
                    'input_datetime': ['CreatedDate', ...],
                    'input_int': [...] 
                    'input_float': [...]
                    }      
    Returns:
      df_y: a DataFrame that stores the output labels
      df_X_float: a DataFrame that stores the float inputs
      df_X_int: a DataFrame that stores the integer inputs
      df_X_cat: a DataFrame that stores the categorical inputs
      df_X_datetime: a DataFrame that stores the datetime inputs
      df_X_bool: a DataFrame that stores the boolean inputs

    """
    # input_cols = metadata['input_features']
    output_cols = metadata['output_label']
    input_text_cols = metadata['input_text']
    input_float_cols = metadata['input_float']
    input_int_cols = metadata['input_int']
    input_cat_cols = metadata['input_categorical']
    input_datetime_cols = metadata['input_datetime']
    input_bool_cols = metadata['input_bool']

    df_y = df.loc[:, output_cols]
    df_X_text = df.loc[:, input_text_cols]
    df_X_float = df.loc[:, input_float_cols]
    df_X_int = df.loc[:, input_int_cols]
    df_X_cat = df.loc[:, input_cat_cols]
    df_X_datetime = df.loc[:, input_datetime_cols]
    df_X_bool = df.loc[:, input_bool_cols]

    return df_y, df_X_text, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool


def encode_datetime(df_X_datetime):
    """Encode the datetime inputs from '2/5/2014 5:31:19 AM' format
        to a numerical number of UTC format.

    Args:
      df_: a DataFrame that only stores the datetime inputs.
        
    Returns:
      X_datetime: a numpy array that contains the encoded datetime inputs.
      datetime_cols: a list that contains the datetime colunms name.   
   
    """
    
    cols = df_X_datetime.columns
    for i in cols:
        df_X_datetime[i] = pd.to_datetime(df_X_datetime[i], utc=True,
                            errors='coerce').astype(int,errors='ignore')
        
    X_datetime = df_X_datetime.to_numpy()
    
    return X_datetime


def encode_bool(df_X_bool):
    """Encode the numerical and boolean inputs.
        
    Args:
      df_X_bool: a DataFrame that stores the boolean inputs
        
    Returns:
      X_bool: a numpy array that contains the encoded boolean inputs.

    """
    X_bool = df_X_bool.astype(int).to_numpy()
    return X_bool


def encode_num(df_X_num):
    """Encode the numerical and boolean inputs.
        
    Args:
      df_X_num: a DataFrame that stores the numerical inputs
        
    Returns:
      X_num: a numpy array that contains the float inputs.
      
    """
    X_num = df_X_num.to_numpy()
    return X_num


def encode_dataset(df, metadata, dv=None, scaler=None, text_token=[None, None]):
    """Encode the raw data in training set.
        
    Args:
      df: a DataFrame that stores the raw data of training set.
      metadata: a dictionary that stores the detail description for features.
        metadata = {'input_features': ['TenantId','CreatedDate', ...]
                    'output_label': ['AR_exchange_06','AR_sharepoint_06', ...]
                    'input_bool': ['HasEXO','HasSPO', ...],
                    'input_categorical': ['CountryCode', 'Languange', ...],
                    'input_datetime': ['CreatedDate', ...],
                    'input_int': [...] 
                    'input_float': [...]
                    }
      dv: a DictVectorizer that is trained on the training set. Default value is None.
      scaler: a StandardScaler that is trained on the training set. Default value is None. 
        
    Returns:
      X_scal: a numpy array that contains the encoded and normalized inputs.
      dv: a DictVectorizer that is trained on the training set.
      scaler: a StandardScaler that is trained on the training set.
      cols_name: a list that contains all of the inputs features after encoding.
      
    """
    
    print('Starting to encode inputs...')
    df_y, df_X_text, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool = separate_input_output_cols(df, metadata)

    y = df_y.to_numpy()
    
    X_list = []
    cols_name = []
    
    if df_X_float.shape[1] > 0:
        X_float = encode_num(df_X_float)
        X_list.append(X_float)
        cols_name += metadata['input_float']

    if df_X_int.shape[1] > 0:
        X_int = encode_num(df_X_int)
        X_list.append(X_int)
        cols_name += metadata['input_int']

    if df_X_bool.shape[1] > 0:
        X_bool = encode_bool(df_X_bool)
        X_list.append(X_bool)
        cols_name += metadata['input_bool']
    
    if df_X_datetime.shape[1] > 0:
        X_datetime = encode_datetime(df_X_datetime)
        X_list.append(X_datetime)
        cols_name += metadata['input_datetime']
    
    ### encode the categorical columns 
    if df_X_cat.shape[1] > 0:
        X_cat_dict = df_X_cat.to_dict(orient='records')

        if dv == None:   
            dv = DictVectorizer(sparse=False)
            X_cat = dv.fit_transform(X_cat_dict)
            
        else:
            X_cat = dv.transform(X_cat_dict)

        X_list.append(X_cat)
        vocab = dv.vocabulary_
        vocab_od = collections.OrderedDict(sorted(vocab.items(), key=lambda x:x[1]))
        cat_encoded_cols = list(vocab_od.keys())
        cols_name += cat_encoded_cols
    
    ### normalize all the inputs except text features
    X_arr = np.concatenate(X_list, axis=1)

    if scaler == None:
        scaler = StandardScaler()
        X_scal = scaler.fit_transform(X_arr)
    else:
        X_scal = scaler.transform(X_arr)

    assert len(cols_name) == X_scal.shape[1]
    print('non-text input data after encoding, the shape is {}'.format(X_scal.shape))
    print('we have {} columns.'.format(len(cols_name)))

    ## encode text columns, encoded text features should not be normalized.
    if df_X_text.shape[1] > 0:
        text_cols = metadata['input_text']
        var = list(zip(text_cols, text_token))
        for col, tok in var:
            if tok == None:
                tok = TfidfVectorizer(max_df=0.9, min_df=10)
                X_encoded = tok.fit_transform(df_X_text[col].values)
                text_token.append(tok)
            else:
                X_encoded = tok.transform(df_X_text[col].values)
            
            X_scal = np.concatenate([X_encoded, X_scal], axis=1)
            
            cols_name.append(col)

    
    return y, X_scal, dv, scaler, text_token, cols_name


if __name__ == '__main__':
    main()