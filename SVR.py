import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import statsmodels.api as sm


file_path = r"C:\Users\Ana\OneDrive - IR INSTITUT ZA VESTACKU INTELIGENCIJU SRBIJE\Desktop\Phd_Projects\PodaciVodaProjekat\WaterQualityPrediction_LOWESS_LSTM_SVR\Podaci\NS-sa 2022.csv"
df = pd.read_csv(file_path)

df.index = pd.to_datetime(df['Datum'] , format = '%m/%d/%Y')

param = 'ep'
# Depending on which feature you want to predict, choose which data you want to extract from the dataset.
# This example uses O2 (mg/l)

# dataset = df[['NS - Proticaj (m3/s)', 'NS - T vode   (°C)', 'NS - O2 (mg/l)']].values
dataset = df[['NS - Proticaj (m3/s)', 'NS - T vode   (°C)', 'NS - Ep. (µS/cm)']].values
dataset=dataset.astype('float32')

day = np.arange(1, dataset.shape[0] + 1)

# LOWESS denoising, frac can be changed

smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,2],is_sorted=True, frac=0.002)
denoised_o2mg_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,1], frac=0.002)
denoised_temp_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))
smoothed=sm.nonparametric.lowess(exog=day, endog=dataset[:,0], frac=0.002)
denoised_proticaj_lowess=np.reshape(smoothed[:, 1],(len(dataset),1))

dataset=np.concatenate((denoised_proticaj_lowess,denoised_temp_lowess,denoised_o2mg_lowess),axis=1)



train_size = int(len(dataset) * 0.70)
val_size= int(len(dataset) * 0.15) 
test_size = len(dataset) - train_size - val_size
train, val, test = dataset[0:train_size,:], dataset[train_size:train_size+val_size,:], dataset[train_size+val_size:len(dataset),:]


# Scaling data to (0,1) range
with open(f"scaler_{param}.pkl", "rb") as file:
    scaler = pickle.load(file)
    
train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# convert an array of values into a dataset matrix
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
trainX, trainY = create_dataset(train_scaled, look_back, steps=5)
testX, testY = create_dataset(test_scaled, look_back, steps=5)
trainX_flat=trainX.reshape(trainX.shape[0],-1)
testX_flat=testX.reshape(testX.shape[0],-1)
#%%
svr1 = SVR()
svr2 = SVR()
svr3 = SVR()
svr4 = SVR()
svr5 = SVR()
parameters={'kernel':['rbf'], 'C':[1, 10,50,100,150,200,500], 'gamma':[0.01,0.05,0.07], 'epsilon':[0.1 ,0.01,0.02]}

# parameters={'kernel':['rbf'], 'C':[1], 'epsilon':[0.1]}


model1=GridSearchCV(svr1,parameters,scoring='neg_root_mean_squared_error',verbose=1, return_train_score=(True))
model2=GridSearchCV(svr2,parameters,scoring='neg_root_mean_squared_error',verbose=1, return_train_score=(True))
model3=GridSearchCV(svr3,parameters,scoring='neg_root_mean_squared_error',verbose=1, return_train_score=(True))
model4=GridSearchCV(svr4,parameters,scoring='neg_root_mean_squared_error',verbose=1, return_train_score=(True))
model5=GridSearchCV(svr5,parameters,scoring='neg_root_mean_squared_error',verbose=1, return_train_score=(True))

model1=model1.fit(trainX_flat, trainY[:,0])

model2=model2.fit(trainX_flat, trainY[:,1])

model3=model3.fit(trainX_flat, trainY[:,2])

model4=model4.fit(trainX_flat, trainY[:,3])

model5=model5.fit(trainX_flat, trainY[:,4])
#%%



# Best parameters are found trough GRIDSearhCV o2
# model1=SVR(kernel='rbf',C=200,epsilon=0.01,gamma=0.01)
# model2=SVR(kernel='rbf',C=200,epsilon=0.01,gamma=0.01)
# model3=SVR(kernel='rbf',C=200,epsilon=0.01,gamma=0.01)
# model4=SVR(kernel='rbf',C=200,epsilon=0.01,gamma=0.01)
# model5=SVR(kernel='rbf',C=200,epsilon=0.01,gamma=0.01)

model1=SVR(kernel='rbf',C=1,epsilon=0.01,gamma=0.01)
model2=SVR(kernel='rbf',C=1,epsilon=0.01,gamma=0.01)
model3=SVR(kernel='rbf',C=1,epsilon=0.01,gamma=0.01)
model4=SVR(kernel='rbf',C=1,epsilon=0.01,gamma=0.01)
model5=SVR(kernel='rbf',C=1,epsilon=0.01,gamma=0.01)

# Separate model for each prediction day
model1=model1.fit(trainX_flat, trainY[:,0])
model2=model2.fit(trainX_flat, trainY[:,1])
model3=model3.fit(trainX_flat, trainY[:,2])
model4=model4.fit(trainX_flat, trainY[:,3])
model5=model5.fit(trainX_flat, trainY[:,4])

# Save separate models

filename = f'model1svr{param}.sav'
pickle.dump(model1, open(filename, 'wb'))
filename = f'model2svr{param}.sav'
pickle.dump(model2, open(filename, 'wb'))
filename = f'model3svr{param}.sav'
pickle.dump(model3, open(filename, 'wb'))
filename = f'model4svr{param}.sav'
pickle.dump(model4, open(filename, 'wb'))
filename = f'model5svr{param}.sav'
pickle.dump(model5, open(filename, 'wb'))

# RESULTS Evaluation