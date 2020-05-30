#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings('ignore')

#function for extending our data of values from previous timestamps
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j-1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

columns=['value']
df_full = pd.DataFrame(columns = columns)
df_full_with_anomalies = pd.DataFrame(columns = ['value', 'timestamp', 'is_anomaly'])

file_path = './data/'


# In[3]:


for i in range(1,2):
    #reading csv
    name = file_path+"real_"+str(i)+'.csv'
    df = pd.read_csv(name, encoding = "ISO-8859-1", header=0)
    df = df.iloc[::-1]
    df.columns = ['timestamp', 'value', 'is_anomaly']
    df_full_with_anomalies = pd.concat([df_full_with_anomalies, df.reset_index()], sort=True, axis=0)
    rolling_with_anomalies = df_full_with_anomalies.value.rolling(window=5)
    df_full_with_anomalies['value'] = rolling_with_anomalies.mean()
    #delete anomalies from dataset
    df_no_anomalies = df[df.is_anomaly != 1]
    df_no_anomalies = df_no_anomalies.iloc[::-1]
    
    #smoothing signal
    rolling = df_no_anomalies.value.rolling(window=5)
    df_no_anomalies['value'] = rolling.mean()
    
    #droping unused columns
    df_no_anomalies.drop(columns = ["is_anomaly", "timestamp"], inplace=True)
    
    #deleting null values
    df_no_anomalies = df_no_anomalies[df_no_anomalies.value.notna()]
    
    #merging files
    df_full = pd.concat([df_full, df_no_anomalies.reset_index()], sort=True, axis=0)
        
df_full = df_full.drop(columns=['index'])
#data normalization
# df_full = (df_full - df_full.mean()) / (df_full.max() - df_full.min())
fig, ax = plt.subplots(num=None, figsize=(30,20), dpi=80, facecolor='w', edgecolor='k')

ax.plot(df_full_with_anomalies.reset_index().value)
ax.plot(df_full_with_anomalies[df_full_with_anomalies.is_anomaly == 1].value, 'r', marker="o")
# fig.show()


# In[4]:


df_full_max = df_full.max()
df_full_min = df_full.min() #equals to 0, so it will not be considered during normalization

df_full_normalized = df_full.apply(lambda x: x/df_full_max-0.5, axis=1)
df_full_normalized
figure(num=None, figsize=(30,20), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_full_normalized.reset_index().value)


# In[5]:


df_full.describe()


# In[6]:


timesteps = 5
X, y = temporalize(X = np.array(df_full), y = np.zeros(len(df_full)), lookback = timesteps)

n_features = 1
X = np.array(X)
X = X.reshape(X.shape[0], timesteps)
Y = X[:, 0]

x_train = X[:1000, :]
x_train = x_train.reshape(x_train.shape[0],timesteps, 1)

x_test = X[1000:1400, :]
x_test = x_test.reshape(x_test.shape[0],timesteps, 1)


x_train.shape,x_test.shape, 


# In[7]:


# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[8]:


y_train = x_train
x_train = np.delete(x_train, 0, axis=0)
y_train = np.delete(y_train, y_train.shape[0]-1, axis=0)


# In[ ]:


lstm_model = model.fit(x_train, y_train , epochs=10, batch_size=5, verbose=0)
#lstm_model = model.load_weights('./weights/weights_21-04-2020_06_52_46.h5')


# In[ ]:


# fit and save the model
import datetime
d = datetime.datetime.today()
model_name = "model_"+d.strftime('%d-%m-%Y_%H_%M_%S')+".h5"
weights_name = "weights_"+d.strftime('%d-%m-%Y_%H_%M_%S')+".h5"

model.save_weights("./weights/"+weights_name)


# In[ ]:


history = lstm_model.history
fig,ax = plt.subplots(figsize=(20,16), dpi=80)
ax.plot(history['loss'], 'b', label="Train", linewidth=2)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.set_title('Model loss', fontsize =16)
ax.legend(loc='upper right')
plt.savefig('model_loss.png') 
plt.show()


# In[ ]:


# demonstrate reconstruction
y_test = x_test
y_test = np.delete(y_test, y_test.shape[0]-1, axis=0)
x_test = np.delete(x_test, 0, axis=0)
yhat = model.predict(x_test, verbose=0)
y_test = y_test.reshape((y_test.shape[0]*y_test.shape[1]),1)[0::5]
yhat = yhat.reshape((yhat.shape[0]*yhat.shape[1]),1)[0::5]


# In[ ]:


fig,ax = plt.subplots(figsize=(20,16), dpi=80)
ax.plot(y_test, 'b', label="Actual", linewidth=3)
ax.plot(yhat, 'y', label="Predicted", linewidth=2)
ax.set_ylabel('Value')
ax.set_xlabel('Timesteps')
ax.set_title('Test Dataset, Real vs Predicted Values', fontsize =16)
ax.legend(loc='upper right')
plt.savefig('test.png') 
plt.show()


# In[ ]:


# demonstrate reconstruction
y_train = x_train
y_train = np.delete(y_train, y_train.shape[0]-1, axis=0)
x_train = np.delete(x_train, 0, axis=0)
yhat_train = model.predict(x_train, verbose=0)
y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1]),1)[0::5]
yhat_train = yhat_train.reshape((yhat_train.shape[0]*yhat_train.shape[1]),1)[0::5]


# In[ ]:


fig,ax = plt.subplots(figsize=(20,16), dpi=80)
ax.plot(y_train, 'b', label="Actual", linewidth=3)
ax.plot(yhat_train, 'y', label="Predicted", linewidth=2)
ax.set_ylabel('Value')
ax.set_xlabel('Timesteps')
ax.set_title('Train Dataset, Real vs Predicted Values', fontsize =16)
ax.legend(loc='upper right')
plt.savefig('train.png') 
plt.show()


# In[ ]:


df_full_with_anomalies[df_full_with_anomalies.is_anomaly ==1]


# In[ ]:


df_full_with_anomalies[df_full_with_anomalies.is_anomaly ==1]
data_to_predict = np.array(df_full_with_anomalies['value'])
df_pred = df_full_with_anomalies.drop(columns=['timestamp', 'is_anomaly', 'index'])
X_with_anomalies, y = temporalize(X = np.array(df_pred), y = np.zeros(len(df_full_with_anomalies)), lookback = timesteps)

n_features = 1
X_with_anomalies = np.array(X_with_anomalies)
X_with_anomalies = X_with_anomalies.reshape(X_with_anomalies.shape[0], timesteps,1)

df_full_with_anomalies_predictions = model.predict(X_with_anomalies)


# In[ ]:


df_values = X_with_anomalies.reshape((X_with_anomalies.shape[0]*X_with_anomalies.shape[1]),1)[0::5]
df_predictions = df_full_with_anomalies_predictions.reshape((df_full_with_anomalies_predictions.shape[0]*df_full_with_anomalies_predictions.shape[1]),1)[0::5]


# In[ ]:


df_is_anomaly = df_full_with_anomalies.drop(columns=['timestamp', 'value', 'index'])
X_is_anomaly, y = temporalize(X = np.array(df_is_anomaly), y = np.zeros(len(df_is_anomaly)), lookback = timesteps)
X_is_anomaly = np.array(X_is_anomaly)
X_is_anomaly = X_is_anomaly.reshape(X_is_anomaly.shape[0], timesteps,1)
X_is_anomaly = X_is_anomaly.reshape((X_is_anomaly.shape[0]*X_is_anomaly.shape[1]),1)[0::5]


# In[ ]:


final_df = pd.DataFrame()
final_df['value'] = df_values.reshape(df_values.shape[0])
final_df['prediction'] = df_predictions.reshape(df_predictions.shape[0])
final_df['is_anomaly'] = X_is_anomaly.reshape(X_is_anomaly.shape[0])
final_df['abs_error'] = np.abs(df_values-df_predictions)
final_df[final_df['is_anomaly'] == 0].abs_error.describe(), final_df[final_df['is_anomaly'] == 1].abs_error.describe()


# In[ ]:


fig,ax = plt.subplots(figsize=(14,16), dpi=80)
ax.plot(final_df['value'], 'b', label="Actual", linewidth=3)
ax.plot(final_df['prediction'], 'y', label="Predicted", linewidth=2)
ax.set_ylabel('Value')
ax.set_xlabel('Timesteps')
ax.set_title('Real vs Predicted Values', fontsize =16)
ax.legend(loc='upper right')
ax.plot(df_full_with_anomalies[df_full_with_anomalies.is_anomaly == 1].value, 'r', marker="o")

plt.show()


# In[ ]:


final_df[final_df.is_anomaly == 1]


# In[ ]:


final_df['abs_error'].mean(), final_df['abs_error'].std()


# In[ ]:


final_df['abs_error'].std()*12


# In[ ]:


final_df['is_anomaly_prediction'] = final_df['abs_error'].apply(lambda x: 0 if np.abs(final_df['abs_error'].mean() - x) < final_df['abs_error'].std()*12 else 1)
final_df[(final_df['is_anomaly'] != final_df['is_anomaly_prediction']) & (final_df['is_anomaly'] == 1)].value.count(), final_df[(final_df['is_anomaly'] != final_df['is_anomaly_prediction']) & (final_df['is_anomaly'] == 0)].value.count()


# In[25]:


# import sklearn.preprocessing as preproc

# fig,ax = plt.subplots(figsize=(14,5), dpi=80)

# anomalies = final_df[(final_df.is_anomaly == 1)]
# no_anomalies = final_df[(final_df.is_anomaly == 0)]

# n_of_bins = 100

# anomalies['quantile'] = preproc.KBinsDiscretizer(n_bins=n_of_bins, encode='ordinal', strategy='quantile').fit_transform(anomalies.abs_error.values.reshape(-1, 1))
# no_anomalies['quantile'] = preproc.KBinsDiscretizer(n_bins=n_of_bins, encode='ordinal', strategy='quantile').fit_transform(no_anomalies.abs_error.values.reshape(-1, 1))

# ax.plot(anomalies.groupby(anomalies['quantile']).abs_error.mean(), label="Anomalia", linewidth=2)
# ax.plot(no_anomalies.groupby(no_anomalies['quantile']).abs_error.mean(), 'r', label="No Anomaly", linewidth=2)

# ax.set_ylabel('Abs Error')
# ax.set_xlabel('Quantile')
# ax.set_title('Quantiled Abs Error for Anomalies and No Anomalies', fontsize =16)
# ax.legend(loc='upper left')
# plt.show()

