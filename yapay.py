# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:22:50 2019

@author: yigit
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#from sklearn.cross_validation import  train_test_split eski modul
from sklearn.model_selection import train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis



start = dt.datetime(2014, 5, 1)
end = dt.datetime(2018, 5, 1) # datatime ile tarih aralıkları belirlendi.

start_test = dt.datetime(2018, 5, 1)
end_test = dt.datetime.now()

data = web.DataReader("AAPL", 'yahoo', start, end) # başlangıç ve bitiş tarihleri arasındaki datalar yahoo ile çekiliyor.
data_test = web.DataReader("AAPL", 'yahoo', start_test, end_test)
data.to_csv("C:/Users/yigit/Desktop/yapay/aapl.csv") # dataframe csv uzantılı dosya olarak kayıt ediliyor.
data_test.to_csv("C:/Users/yigit/Desktop/yapay/aapl_test.csv")
# %% 

plt.plot(data['Close']) # close değerlerini plot ediliyor

plt.show()

# %% RSI indikatör def

def RSI(candles, length):
       delta = candles['Close'].diff()
       delta = delta[1:]
       up, down = delta.copy(), delta.copy()
       up[up < 0.0] = 0.0
       down[down > 0.0] = 0.0
       roll_up1 = up.ewm(com=(length-1), min_periods=length).mean()
       roll_down1 = down.abs().ewm(com=(length-1), min_periods=length).mean()
       RS1 = roll_up1 / roll_down1
       RSI1 = 100.0 - (100.0 / (1.0 + RS1))
       return RSI1

# %% 


rsi = RSI(data, 14)
plt.plot(rsi)
plt.show() 

print(type(rsi))

rsi=rsi.to_frame() # series to dataframe
rsi=rsi.ewm(span=20,adjust=False).mean() # hareketli ortalama yaparak sinyaldeki gürültüler azaltılıyor
rsi=rsi.dropna()
a=data['Close'].ewm(span=20, adjust=False).mean() # rsi için yapılan gürültü azaltmayı data['Close'] içinde yapıyorum

a=a.to_frame()     

rsi_test=RSI(data_test,14)

rsi_test=rsi_test.to_frame()
rsi_test=rsi_test.dropna()
rsi_test=rsi_test.ewm(span=20,adjust=False).mean()

a_test=data_test['Close'].ewm(span=20, adjust=False).mean() 
a_test=a_test.to_frame()



print(a.shape, "\n", rsi.shape)
# %% 

plt.plot(rsi)
plt.show() 

plt.plot(a)
plt.show()

plt.plot(data['Close'])
plt.show()

# %% 

# verileri  0 ile1 1 arasında sıkıştırıyorum. 
# rsi indikatörü 0 ile 100 arasında olduğu için belki 0-1 arasında işlem yapmak gerekmeyebilir? sonuçta birbirleri arasında çok büyük farklar olmayacaktır..

scaler=MinMaxScaler(feature_range=(0,1)) # normalizasyon amaçlı

a=scaler.fit_transform(a)

rsi=scaler.fit_transform(rsi)

a_test=scaler.fit_transform(a_test)

rsi_test=scaler.fit_transform(rsi_test)

# data['Close'] ve rsi 0-1 arasında değerler aldı

print(a.shape,"\n",rsi.shape)

# %% 

x_train_a = []
y_train_a = []
timestamp = 60
length = len(a)
for i in range(timestamp, length):
    x_train_a.append(a[i-timestamp:i, 0])
    y_train_a.append(a[i, 0])
    
x_train_a = np.array(x_train_a)
y_train_a = np.array(y_train_a)

print(a,"\n-----------\n",x_train_a,"\n------------\n",y_train_a)
print("\n",a.shape,"\n",x_train_a.shape,"\n",y_train_a.shape)


x_test_a = []

length = len(a_test)
for i in range(timestamp, length):
    x_test_a.append(a_test[i-timestamp:i, 0])
    
x_test_a = np.array(x_test_a)




# %% reshape

x_train_a=np.reshape(x_train_a,(x_train_a.shape[0],x_train_a.shape[1],1))

x_test_a=np.reshape(x_test_a,(x_test_a.shape[0],x_test_a.shape[1],1))

# %% rsi için
x_train_rsi = []
y_train_rsi = []
timestamp = 60
length = len(rsi)
for i in range(timestamp, length):
    x_train_rsi.append(rsi[i-timestamp:i, 0])
    y_train_rsi.append(rsi[i, 0])
    
x_train_rsi = np.array(x_train_rsi)
y_train_rsi = np.array(y_train_rsi)

print(rsi,"\n-----------\n",x_train_rsi,"\n------------\n",y_train_rsi)
print("\n",rsi.shape,"\n",x_train_rsi.shape,"\n",y_train_rsi.shape)

x_test_rsi = []

length = len(rsi_test)
for i in range(timestamp, length):
    x_test_rsi.append(rsi_test[i-timestamp:i, 0])
    
x_test_rsi = np.array(x_test_rsi)


# %% 
x_train_rsi=np.reshape(x_train_rsi,(x_train_rsi.shape[0],x_train_rsi.shape[1],1))

x_test_rsi=np.reshape(x_test_rsi,(x_test_rsi.shape[0],x_test_rsi.shape[1],1))

# %% eğitim


model = Sequential()

model.add(LSTM(units = 25, return_sequences = True, input_shape = (x_train_a.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 25, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 25, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 25, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# %% hareketli ortalama

model.fit(x_train_a, y_train_a, epochs = 25, batch_size = 32)




# %% 

model_rsi = Sequential()

model_rsi.add(LSTM(units = 25, return_sequences = True, input_shape = (x_train_a.shape[1], 1)))
model_rsi.add(Dropout(0.2))

model_rsi.add(LSTM(units = 25, return_sequences = True))
model_rsi.add(Dropout(0.2))

model_rsi.add(LSTM(units = 25, return_sequences = True))
model_rsi.add(Dropout(0.2))

model_rsi.add(LSTM(units = 25, return_sequences = False))
model_rsi.add(Dropout(0.2))

model_rsi.add(Dense(units = 1))
model_rsi.compile(optimizer = 'adam', loss = 'mean_squared_error')
# %% rsi ile 


model_rsi.fit(x_train_rsi, y_train_rsi, epochs = 25, batch_size = 32)









 