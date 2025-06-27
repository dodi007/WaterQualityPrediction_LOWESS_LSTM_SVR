# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:08:53 2025

@author: Ana
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import keras 

file_path = r"C:\Users\Ana\OneDrive - IR INSTITUT ZA VESTACKU INTELIGENCIJU SRBIJE\Desktop\Phd_Projects\PodaciVodaProjekat\WaterQualityPrediction_LOWESS_LSTM_SVR\Podaci\StanicaNoviSad2024.csv"
df = pd.read_csv(file_path)

df.index = pd.to_datetime(df['Datum'] , format = '%m/%d/%Y')

# Depending on which feature you want to predict, choose which data you want to extract from the dataset.
# This example uses O2 (mg/l)

param = 'ep'

if param == 'ep':
    dataset = df[['NS - Proticaj (m3/s)', 'NS - T vode   (°C)', 'NS - Ep. (µS/cm)']].values
else:
    dataset = df[['NS - Proticaj (m3/s)', 'NS - T vode   (°C)', 'NS - O2 (mg/l)']].values

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

with open(f"scaler_{param}.pkl", "rb") as file:
    scaler = pickle.load(file)
# Scaling data to (0,1) range

train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)
new_test_scaled = scaler.transform(dataset)

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

test_newX, test_newY = create_dataset(new_test_scaled,dataset, look_back, steps)
@keras.saving.register_keras_serializable()
class PinballLoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.6, reduction=tf.keras.losses.Reduction.AUTO, name='pinball_loss'):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - tf.squeeze(y_pred, axis=-1)

        return tf.reduce_mean(tf.maximum(self.quantile * error, (self.quantile - 1) * error))

model=tf.keras.models.load_model(f'LSTM_model_{param}.keras')
lstm_forecast = model.predict(test_newX)
#%%
final_output = np.zeros((lstm_forecast.shape[0], lstm_forecast.shape[1]))
for i in range(lstm_forecast.shape[1]):
    
    dummy_array = np.zeros((lstm_forecast.shape[0], scaler.n_features_in_))
    dummy_array[:, 2] = lstm_forecast[:, i].reshape(-1)
    inverse_transformed = scaler.inverse_transform(dummy_array)
    final_output[:,i] = inverse_transformed[:,2]

lstm_forecast = final_output
#%%
testX, testY = create_dataset(dataset, dataset,  look_back, steps=5)
import matplotlib.pyplot as plt
plt.plot(testY[:,0], label='Measured')
plt.plot(lstm_forecast[:,0], label='Predicted', lw=3)
plt.legend()
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error

for i in range (steps):
    
    print(f'MAE - mean absolute error for step {i+1} - ', mean_absolute_error(testY[:,i], lstm_forecast[:,i]))
    print(f'MAPE - mean absolute percentage error for step {i+1} - ', mean_absolute_percentage_error(testY[:,i], lstm_forecast[:,i]))
    print(f'R2 - coefficient of determination for step {i+1} - ', r2_score(testY[:,i], lstm_forecast[:,i]))
    print(f'RMSE - root mean squared error for step {i+1} - ', np.sqrt(mean_squared_error(testY[:,i], lstm_forecast[:,i])))
    print(f'Max error - for step {i+1} ', np.max(np.abs(testY[:,i] - lstm_forecast[:,i])))
    
results = []

for i in range(steps):
    mae = mean_absolute_error(testY[:,i], lstm_forecast[:,i])
    mape = mean_absolute_percentage_error(testY[:,i], lstm_forecast[:,i])
    r2 = r2_score(testY[:,i], lstm_forecast[:,i])
    rmse = np.sqrt(mean_squared_error(testY[:,i], lstm_forecast[:,i]))
    max_err = np.max(np.abs(testY[:,i] - lstm_forecast[:,i]))
    
    results.append({
        'Step': i+1,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'RMSE': rmse,
        'Max_Error': max_err
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(f'lstm_evaluation_metrics_{param}.csv', index=False)

#%%  
import matplotlib
matplotlib.rc('font', size=14)
import numpy as np
import matplotlib.pyplot as plt
import os

# parameter = 'Dissolved oxygen [mg/l]'
# param = 'ep'
parameter = 'Conductivity [µS/cm]'
timestamps = df.index[10:-5]
# Scatter plot for all prediction days
plt.figure(figsize=(10, 6))
for i in range(steps):
    plt.scatter(testY[:, i], lstm_forecast[:, i], label=f'Day {i+1}', alpha=0.6)
plt.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'k--', lw=2)  # Diagonal reference line
plt.xlabel(f"Actual Values for {parameter}")
plt.ylabel(f"Predicted Values for {parameter}")
plt.title(f"LSTM - Scatter Plot - {parameter}")
plt.legend()
plot_filename = os.path.join(os.getcwd(), f'LSTM_scatter_{param}.png')
plt.savefig(plot_filename, format='png', bbox_inches='tight')
plt.show()

# Time series plot for all prediction days
plt.figure(figsize=(12, 7))
for i in range(steps):
    if i==0:
        plt.plot(timestamps,testY[:, i], label='Observed', linestyle='-')
    plt.plot(timestamps,lstm_forecast[:, i], label=f'Days Ahead {i+1}', linestyle = '--')
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.ylabel(f"{parameter}")
plt.title(f"LSTM - Time Series Plot - {parameter}")
plot_filename = os.path.join(os.getcwd(), f'LSTM_time_{param}.png')
plt.legend()
plt.savefig(plot_filename, format='png', bbox_inches='tight')


plt.show()