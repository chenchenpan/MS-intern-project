import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import json
import random
from sklearn.metrics import mean_squared_error
import time
from tensorflow.keras.models import model_from_json
# from tensorflow.keras import layers
from tensorflow.keras import optimizers

Xdev = np.load('data/Xdev.npy')
ydev = np.load('data/ydev.npy')
n_example, n_features = Xdev.shape

model_n = 28
opt = optimizers.Adam()

# load json and create model
model_path = 'results/NNmodel_{}_with_1.0data.json'.format(model_n)
print(model_path)
weights_path = 'results/NNmodel_{}_with_1.0data_weights.hdf5'.format(model_n)

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

randsample = 10

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

np.save('results/permut_dev_results.npy', permutsample)
