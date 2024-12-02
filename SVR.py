import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
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

#%% convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1,steps=1):
     dataX, dataY = [], []
     output=dataset[:,2].tolist()
     for i in range(len(dataset)-look_back-steps):
         a = dataset[i:(i+look_back), :]
         dataX.append(a)
         # dataY.append(dataset[i + look_back, 2])
         dataY.append(output[i+look_back:i+look_back+steps])
     return np.array(dataX), np.array(dataY)
     
        
# reshape into X=t and Y=t+1
look_back = 10
steps=5
trainX, trainY = create_dataset(train, look_back, steps=5)
testX, testY = create_dataset(test, look_back, steps=5)
trainX_flat=trainX.reshape(trainX.shape[0],-1)
testX_flat=testX.reshape(testX.shape[0],-1)

# Best parameters are found trough GRIDSearhCV
model1=SVR(kernel='rbf',C=100,epsilon=0.01,gamma=0.05)
model2=SVR(kernel='rbf',C=100,epsilon=0.01,gamma=0.05)
model3=SVR(kernel='rbf',C=100,epsilon=0.01,gamma=0.05)
model4=SVR(kernel='rbf',C=100,epsilon=0.01,gamma=0.05)
model5=SVR(kernel='rbf',C=100,epsilon=0.01,gamma=0.05)

# Separate model for each prediction day
model1=model1.fit(trainX_flat, trainY[:,0])
model2=model2.fit(trainX_flat, trainY[:,1])
model3=model3.fit(trainX_flat, trainY[:,2])
model4=model4.fit(trainX_flat, trainY[:,3])
model5=model5.fit(trainX_flat, trainY[:,4])

# Save separate models

filename = 'model1svro2.sav'
pickle.dump(model1, open(filename, 'wb'))
filename = 'model2svro2.sav'
pickle.dump(model2, open(filename, 'wb'))
filename = 'model3svro2.sav'
pickle.dump(model3, open(filename, 'wb'))
filename = 'model4svro2.sav'
pickle.dump(model4, open(filename, 'wb'))
filename = 'model5svro2.sav'
pickle.dump(model5, open(filename, 'wb'))

# RESULTS Evaluation