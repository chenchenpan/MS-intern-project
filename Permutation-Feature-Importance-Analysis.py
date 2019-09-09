import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import json
import random
import argparse
import os
from sklearn.metrics import mean_squared_error
import time
from tensorflow.keras.models import model_from_json
# from tensorflow.keras import layers
from tensorflow.keras import optimizers

def main():

    parser = argparse.ArgumentParser()

    # load the dev data
    parser.add_argument('--data_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/encoded_data',
        help=('directory to load the dev data.'))
    
    # load the model and experiments results
    parser.add_argument('--experiment_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/results',
        help=('directory to load the model and experiment results.'))
    
    # parameter for saving results
    parser.add_argument('--output_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/results',
        help=('directory to store the results.'))
    
    # parameter for number of trials
    parser.add_argument('--trial_num', type=int,
        default=10,
        help=('how many times you want to permute?'))


    args = parser.parse_args()

    Xdev_path = os.path.join(args.data_dir, 'Xdev.npy')
    Xdev = np.load(Xdev_path)
    ydev_path = os.path.join(args.data_dir, 'ydev.npy')
    ydev = np.load(ydev_path)

    experi_path = os.path.join(args.experiment_dir, 'NN_experiments_with_1.0data.csv')
    experi = pd.read_csv(experi_path)

    experi = experi.sort_values('best_dev_mse', axis=0)
    best = experi.iloc[:1,:]
    model_n = best.index[0] 

    hyp_str = best['hyperparam'].to_list()[0]
    hyp = hyp_str.replace("'", "\"")
    hyp_params = json.loads(hyp)
    print(hyp_params)

    opt = optimizers.Adam()

    # load the best model
    model_path = os.path.join(args.experiment_dir,'NNmodel_{}_with_1.0data.json'.format(model_n))
    print(model_path)
    weights_path = os.path.join(args.experiment_dir,'NNmodel_{}_with_1.0data_weights.hdf5'.format(model_n))
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")

    # evaluate loaded model 

    loaded_model.compile(loss='mean_squared_error', optimizer=opt,
                        metrics=['mean_squared_error'])

    pred_dev = loaded_model.predict(Xdev)
    dev_mse = mean_squared_error(ydev, pred_dev)
    dev_mse_path = os.path.join(args.output_dir, 'dev_pred_mse.npy')
    np.save(dev_mse_path, dev_mse)

    
    randsample = args.trial_num

    permut_arr = permute(loaded_model, randsample, Xdev, ydev)

    save_path = os.path.join(args.output_dir, 'permut_dev_results.npy')
    np.save(save_path, permut_arr)
    
    
def permute(loaded_model, randsample, Xdev, ydev):
    n_example, n_features = Xdev.shape

    permutsample = np.zeros((randsample, n_features))
    for trying in range(randsample):
        t1 = time.time()
        print('This is sample {}'.format(trying))
        for i in range(n_features):
            permut_X = np.zeros(np.shape(Xdev))
            permut_X[:] = Xdev
            permut_X[:,-i] = np.random.permutation(Xdev[:,-i])
            permut_pred = loaded_model.predict(permut_X)
            permut_mse = mean_squared_error(ydev, permut_pred)
            permutsample[trying, -i] = permut_mse
        t2 = time.time()
        print('Use time in {} seconds'.format(t2-t1))
        print('*' * 20)

    return permutsample


# permutsample = np.zeros((randsample, n_features))
# for trying in range(randsample):
#     t1 = time.time()
#     print('This is sample {}'.format(trying))
#     for i in range(n_features):
#         permut_X = np.zeros(np.shape(Xdev))
#         permut_X[:] = Xdev
#         permut_X[:,-i] = np.random.permutation(Xdev[:,-i])
#         permut_pred = loaded_model.predict(permut_X)
#         permut_mse = mean_squared_error(ydev, permut_pred)
#         permutsample[trying, -i] = permut_mse
#     t2 = time.time()
#     print('Use time in {} seconds'.format(t2-t1))
#     print('*' * 20)

if __name__ == '__main__':
    main()


