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
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
import time
# from sklearn.manifold import TSNE
# import joblib


Xtrain = np.load('encoded_data_clip_fast/Xtrain.npy')
Xdev = np.load('encoded_data_clip_fast/Xdev.npy')
# Xtest = np.load('encoded_data/Xtest.npy')
ytrain = np.load('encoded_data_clip_fast/ytrain.npy')
ydev = np.load('encoded_data_clip_fast/ydev.npy')
# ytest = np.load('encoded_data/ytest.npy')

data_size = [0.001, 0.002, 0.004, 0.008, 0.016, 0.02, 0.03, 0.04, 0.08, 0.10, 0.15, 0.20]

## KNN model
knn_results = []
full_size = Xtrain.shape[0]
print('full training size is {}'.format(full_size))
# Xdev = Xdev[:100,:]
# ydev = ydev[:100,:]

for i in data_size:

    t1 = time.time()
    results = {}
    
    selected_size = int(full_size * i)
    print('selected training size is {}'.format(selected_size))
    
    Xtrain_selected = Xtrain[:selected_size,:]
    ytrain_selected = ytrain[:selected_size,:]

    model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5, n_jobs=2))
    model.fit(Xtrain_selected, ytrain_selected)
    pred_dev = model.predict(Xdev)
    mse_dev = mean_squared_error(ydev, pred_dev)
    
    results['data_size'] = i
    results['Dev_mse'] = mse_dev
    
    knn_results.append(results)

    t2 = time.time()
    print('used time in {} secondes.'.format(t2-t1))

knn_df = pd.DataFrame(knn_results)
knn_df.to_csv('results_clip_fast/knn_results_with_5_neighbors.csv', index=False)