import numpy as np
import pandas as pd
import json
import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def main():

    parser = argparse.ArgumentParser()

    # pick up the size for training
    parser.add_argument('--training_size', type=float,
        default=1.0,
        help=('how much percentage of training data will be used'))
    
    # select data source folder
    parser.add_argument('--data_dir', type=str,
        default='/data/home/t-chepan/projects/MS-intern-project/encoded_data',
        help=('directory to load the encoded data.'))
    
    # parameter for saving models
    parser.add_argument('--output_dir', type=str,
        default='/data/home/t-chepan/projects/MS-intern-project/results',
        help=('directory to store final and intermediate '
            'results and models.'))
    
    # parameter for number of trials
    parser.add_argument('--trial_num', type=int,
        default=10,
        help=('how many models you want to try?'))


    args = parser.parse_args()


    print('Starting to load data...')
    Xtrain_path = os.path.join(args.data_dir, 'Xtrain.npy')
    Xtrain = np.load(Xtrain_path)

    # ### check encoded data has no nan
    # if np.any(Xtrain == nan)

    full_size = Xtrain.shape[0]
    selected_size = int(full_size * args.training_size)

    Xtrain = Xtrain[:selected_size,:]

    ytrain_path = os.path.join(args.data_dir, 'ytrain.npy')
    ytrain = np.load(ytrain_path) 
    ytrain = ytrain[:selected_size,:]

    Xdev_path = os.path.join(args.data_dir, 'Xdev.npy')
    Xdev = np.load(Xdev_path)
    ydev_path = os.path.join(args.data_dir, 'ydev.npy')
    ydev = np.load(ydev_path)

    n_features = Xtrain.shape[1]
    n_outputs = ytrain.shape[1]

    EPOCHS = 200
    PATIENCE = 5

    def train_model(hyp_params, max_n_epoch=EPOCHS, patience=PATIENCE, name='model'):
        inputs = keras.Input(shape=(n_features,), name='input_features')
        x = layers.Dense(hyp_params['fc_hidden_size'], activation='relu')(inputs)

        for _ in range(hyp_params['n_fc_layers']-1):
            x = layers.Dense(hyp_params['fc_hidden_size'], activation='relu')(x)
            # x = layers.Dropout(hyp_params['dropout_rate'])(x)

        outputs = layers.Dense(n_outputs)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        if hyp_params['opt'] == 'adam':
            opt = optimizers.Adam(lr=hyp_params['lr'])
        elif hyp_params['opt'] == 'rmsprop':
            opt = optimizers.RMSprop(lr=hyp_params['lr'])
        elif hyp_params['opt'] == 'sgd':
            opt = optimizers.SGD(lr=hyp_params['lr'])
        else:
            raise ValueError('Unknown optimizer: {}'.format(hyp_params['opt']))

        
        model.compile(loss='mean_squared_error',
        optimizer=opt,
        metrics=['mean_squared_error'])

        model_json = model.to_json()
        model_path = os.path.join(args.output_dir, '{}.json'.format(name))
        with open(model_path, 'w') as json_file:
            json_file.write(model_json)

        weights_path = os.path.join(args.output_dir, '{}_weights.hdf5'.format(name))
        checkpointer = ModelCheckpoint(filepath=weights_path,
            monitor='val_loss',
            verbose=0, save_best_only=True)

        early_stop = EarlyStopping(monitor='val_loss', patience=patience)

        callbacks_list = [early_stop, checkpointer]

        hist = model.fit(Xtrain, ytrain,
            batch_size=64,
            epochs=max_n_epoch,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=(Xdev, ydev))

        return hist, model
    
    
    experiments = []
    for i in range(args.trial_num):
        hyp_params = {'n_fc_layers': 3,
            # 'dropout_rate': 0.5,
            'fc_hidden_size': 64,
            'opt': 'adam',
            'lr': 0.0003338803745438395}

        hyp_params['n_fc_layers'] = np.random.randint(2, 10)
        # hyp_params['dropout_rate'] = np.random.uniform(0.0, 0.5)
        hyp_params['fc_hidden_size'] = [64, 128, 256][np.random.randint(0, 3)]
        hyp_params['lr'] = 10 ** (np.random.uniform(-2, -4))
        # hyp_params['opt'] = ['adam', 'rmsprop', 'sgd'][np.random.randint(0,3)]
        hyp_params['opt'] = 'adam'
        print(i)
        print(hyp_params)
        hist, _ = train_model(hyp_params, name="NNmodel_{}_with_{}data".format(i, args.training_size))
        experiments.append({'hyperparam': hyp_params, 'history': hist.history,  
                            'best_dev_mse': min(hist.history['val_mean_squared_error'])})
        print(hist.history['val_mean_squared_error'])

        df = pd.DataFrame(experiments)
        df_path = os.path.join(args.output_dir, 'NN_experiments_with_{}data.csv'.format(args.training_size))
        df.to_csv(df_path, index=False)


if __name__ == '__main__':
    main()


