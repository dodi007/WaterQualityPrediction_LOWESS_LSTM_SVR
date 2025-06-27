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
import keras


file_path = r"C:\Users\Ana\OneDrive - IR INSTITUT ZA VESTACKU INTELIGENCIJU SRBIJE\Desktop\Phd_Projects\PodaciVodaProjekat\WaterQualityPrediction_LOWESS_LSTM_SVR\Podaci\NS-sa 2022.csv"
df = pd.read_csv(file_path)

df.index = pd.to_datetime(df['Datum'] , format = '%m/%d/%Y')

# Depending on which feature you want to predict, choose which data you want to extract from the dataset.
# This example uses O2 (mg/l)

# dataset = df[['NS - Proticaj (m3/s)', 'NS - T vode   (°C)', 'NS - O2 (mg/l)']].values
dataset = df[['NS - Proticaj (m3/s)', 'NS - T vode   (°C)', 'NS - Ep. (µS/cm)']].values
dataset=dataset.astype('float32')

day = np.arange(1, dataset.shape[0] + 1)

# LOWESS denoising, frac can be changed

import statsmodels.api as sm
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,2],is_sorted=True, frac=0.002)
denoised_o2mg_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,1], frac=0.002)
denoised_temp_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,0], frac=0.002)
denoised_proticaj_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))

dataset=np.concatenate((denoised_proticaj_lowess,denoised_temp_lowess,denoised_o2mg_lowess),axis=1)

n_features=dataset.shape[1]

#  Divide data into the train, val, and test set

train_size = int(len(dataset) * 0.70)
val_size= int(len(dataset) * 0.15) 
test_size = len(dataset) - train_size - val_size
train, val, test = dataset[0:train_size,:], dataset[train_size:train_size+val_size,:], dataset[train_size+val_size:len(dataset),:]

# Scaling data to (0,1) range
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

# Apply the same scaler to the validation and test sets
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)
with open('scaler_ep.pkl', 'wb') as f:
    pickle.dump(scaler,f)
    
@keras.saving.register_keras_serializable()
class PinballLoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.6, reduction=tf.keras.losses.Reduction.AUTO, name='pinball_loss'):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - tf.squeeze(y_pred, axis=-1)

        return tf.reduce_mean(tf.maximum(self.quantile * error, (self.quantile - 1) * error))

def create_dataset(dataset_scaled, dataset_orig, look_back=1,steps=1):
     # Function to create dataset compatible to LSTM input shape
     dataX, dataY = [], []
     output=dataset_scaled[:,2].tolist()
     for i in range(len(dataset_scaled)-look_back-steps):
         a = dataset_scaled[i:(i+look_back), :]
         dataX.append(a)
         dataY.append(output[i+look_back:i+look_back+steps])
     return np.array(dataX), np.array(dataY)

look_back = 10
steps=5
trainX, trainY = create_dataset(train_scaled, train, look_back, steps=5)
testX, testY = create_dataset(test_scaled, test,  look_back, steps=5)

valX, valY = create_dataset(val_scaled, val, look_back,steps)
n_steps_in, n_steps_out = 10, 5

# Bayesian Hyperparameter Optimization

space = {
    'num_layers': hp.choice('num_layers', [1,2, 3]),
    'units_layer_1': hp.choice('units_layer_1', [32, 64, 128]),
    'units_layer_2': hp.choice('units_layer_2', [32, 64, 128]),
    'units_layer_3': hp.choice('units_layer_3', [32, 64, 128]),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'activation_layer_1': hp.choice('activation_layer_1', ['relu', 'sigmoid', 'tanh']),
    'activation_layer_2': hp.choice('activation_layer_2', ['relu', 'sigmoid', 'tanh']),
    'activation_layer_3': hp.choice('activation_layer_3', ['relu', 'sigmoid', 'tanh']),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'optimizer': hp.choice('optimizer', [
        {
            'type': 'adam',
            'lr': hp.loguniform('lr_a', np.log(0.0001), np.log(0.01))
        },
        {
            'type': 'sgd',
            'lr': hp.loguniform('lr_s', np.log(0.0001), np.log(0.01)),
            'momentum': hp.uniform('momentum', 0, 1)
        }]),
    'batch_size': hp.choice('batch_size',[16,32,64])   
}



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

    pinball_loss = PinballLoss(quantile=0.65)
    model.compile(loss=pinball_loss, optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.001, restore_best_weights=True)
    
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=200, batch_size=params['batch_size'], callbacks=[early_stopping], verbose=2)
    loss, accuracy = model.evaluate(valX, valY, verbose=0)
    return {'loss': loss, 'status': STATUS_OK}


# Find best Hyperparameters
trials = Trials()
best = fmin(fn=lstm_model_optim, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

best_params = space_eval(space, best)

print("Best hyperparameters: ", best_params)

with open('best_hyperparams_lstm_ep.pkl', 'wb') as f:
    pickle.dump(best_params,f)

with open('best_hyperparams_lstm_ep.pkl', 'rb') as f:
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
    if optimizer['type'] == 'adam':
        optimizer = Adam(learning_rate=optimizer['lr'])
    else:  # SGD
        optimizer = SGD(learning_rate=optimizer['lr'], momentum=optimizer['momentum'])

    pinball_loss = PinballLoss(quantile=0.65)
    model.compile(loss=pinball_loss, optimizer=optimizer, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001, restore_best_weights=True)
    
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=500, batch_size=params['batch_size'], callbacks=[early_stopping], verbose=2)
    loss, accuracy = model.evaluate(valX, valY, verbose=2)
    return model

# Train LSTM model with best Hyperparameters
model = lstm(best)
model.save('LSTM_model_ep.keras')

