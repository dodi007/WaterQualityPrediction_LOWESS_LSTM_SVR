import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pickle

file_path = 'path_to_dataset'
df = pd.read_csv(file_path)

df.index = pd.to_datetime(df['Datum'] , format = '%m/%d/%Y')

# Depending on which feature you want to predict, choose which data you want to extract from the dataset.
# This example uses O2 (mg/l)

dataset = df[['Proticaj (m3/s)', 'T vode   (°C)', 'O2 (mg/l)']].values
dataset=dataset.astype('float32')

day = np.arange(1, dataset.shape[0] + 1)

# LOWESS denoising, frac can be changed

smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,2],is_sorted=True, frac=0.004)
denoised_o2mg_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,1], frac=0.004)
denoised_temp_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,0], frac=0.004)
denoised_proticaj_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))

dataset=np.concatenate((denoised_proticaj_lowess,denoised_temp_lowess,denoised_o2mg_lowess),axis=1)

# Scaling data to (0,1) range

scaler= MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler,f)
n_features=dataset.shape[1]

#  Divide data into the train, val, and test set

train_size = int(len(dataset) * 0.70)
val_size= int(len(dataset) * 0.15) 
test_size = len(dataset) - train_size - val_size
train, val, test = dataset[0:train_size,:], dataset[train_size:train_size+val_size,:], dataset[train_size+val_size:len(dataset),:]


def create_dataset(dataset, look_back=1,steps=1):
     # Function to create dataset compatible to LSTM input shape
     dataX, dataY = [], []
     output=dataset[:,2].tolist()
     for i in range(len(dataset)-look_back-steps):
         a = dataset[i:(i+look_back), :]
         dataX.append(a)
         dataY.append(output[i+look_back:i+look_back+steps])
     return np.array(dataX), np.array(dataY)

look_back = 10
steps=5
trainX, trainY = create_dataset(train, look_back, steps=5)
testX, testY = create_dataset(test, look_back, steps=5)
valX, valY = create_dataset(val,look_back,steps)

# Bayesian Hyperparameter Optimization

space = {
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'units_layer_1': hp.choice('units_layer_1', [16, 32, 64]),
    'units_layer_2': hp.choice('units_layer_2', [16, 32, 64]),
    'units_layer_3': hp.choice('units_layer_3', [16, 32, 64]),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'activation_layer_1': hp.choice('activation_layer_1', ['relu', 'sigmoid', 'tanh']),
    'activation_layer_2': hp.choice('activation_layer_2', ['relu', 'sigmoid', 'tanh']),
    'activation_layer_3': hp.choice('activation_layer_3', ['relu', 'sigmoid', 'tanh']),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
    'optimizer': hp.choice('optimizer', [
        {
            'type': 'adam',
            'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01))
        },
        {
            'type': 'sgd',
            'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
            'momentum': hp.uniform('momentum', 0, 1)
        }]),
    'batch_size': hp.choice('batch_size',[16,32,64])   
}

n_steps_in, n_steps_out = 10, 5

def lstm_model_optim(params):
    model = Sequential()
    
    # Add multiple LSTM layers
    for i in range(params['num_layers']):
        # Choose the number of units based on the layer
        units = params[f'units_layer_{i+1}']
        activation = params[f'activation_layer_{i+1}']
        # Return sequences for all layers but the last one
        return_sequences = i < params['num_layers'] - 1
        model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=(trainX.shape[1], trainX.shape[2]),activation=activation))
        model.add(Dropout(params['dropout']))

    model.add(Dense(n_steps_out, activation=params['activation']))  # 2 outputs for each day ahead
    model.add(Reshape((n_steps_out, 1)))

    optimizer_params = params['optimizer']
    if optimizer_params['type'] == 'adam':
        optimizer = Adam(learning_rate=optimizer_params['lr'])
    else:  # SGD
        optimizer = SGD(learning_rate=optimizer_params['lr'], momentum=optimizer_params['momentum'])

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.001, restore_best_weights=True)
    
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=200, batch_size=params['batch_size'], callbacks=[early_stopping], verbose=2)
    loss, accuracy = model.evaluate(valX, valY, verbose=2)
    return {'loss': loss, 'status': STATUS_OK}


# Find best Hyperparameters
trials = Trials()
best = fmin(fn=lstm_model_optim, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

best_params = space_eval(space, best)

print("Best hyperparameters: ", best_params)

with open('best_hyperparams_lstm_do.pkl', 'wb') as f:
    pickle.dump(best_params,f)

with open('best_hyperparams_lstm_do.pkl', 'rb') as f:
    loaded_best=pickle.load(f)

best=loaded_best

def lstm(params):
    model = Sequential()
    
    # Add multiple LSTM layers
    for i in range(params['num_layers']):
        # Choose the number of units based on the layer
        unitss = params[f'units_layer_{i+1}']
        activationn = params[f'activation_layer_{i+1}']
        # Return sequences for all layers but the last one
        return_sequences = i < params['num_layers'] - 1
        model.add(LSTM(units=unitss, return_sequences=return_sequences, input_shape=(trainX.shape[1], trainX.shape[2]),activation=activationn))
        model.add(Dropout(params['dropout']))

    model.add(Dense(n_steps_out, activation=params['activation']))  # 2 outputs for each day ahead
    model.add(Reshape((n_steps_out,1)))

    optimizer = params['optimizer']
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=optimizer['lr'])
    else:  # SGD
        optimizer = SGD(learning_rate=optimizer['lr'], momentum=optimizer['momentum'])

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001, restore_best_weights=True)
    
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=500, batch_size=params['batch_size'], callbacks=[early_stopping], verbose=2)
    loss, accuracy = model.evaluate(valX, valY, verbose=2)
    return model


# Train LSTM model with best Hyperparameters
model = lstm(best)
model.save('LSTM_model')

#RESULT EVALUTAION
