import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from stockstats import StockDataFrame as Sdf

dt = pd.read_csv('../python for finance/GBPUSD2000_2018.csv', header=0, index_col=0)

stock = Sdf.retype(dt)

### open delta against next 2 day
##stock['open_2_d']
### open price change (in percent) between today and the day before yesterday
### 'r' stands for rate.
##stock['open_-2_r']
### CR indicator, including 5, 10, 20 days moving average
##stock['cr']
##stock['cr-ma1']
##stock['cr-ma2']
##stock['cr-ma3']
### KDJ, default to 9 days
##stock['kdjk']
##stock['kdjd']
##stock['kdjj']
### 2 days simple moving average on open price
##stock['open_2_sma']
### MACD
##stock['macd']
### MACD signal line
##stock['macds']
### MACD histogram
##stock['macdh']
### bolling, including upper band and lower band
##stock['boll']
##stock['boll_ub']
##stock['boll_lb']
### 6 days RSI
##stock['rsi_6']
### 12 days RSI
##stock['rsi_12']
### 10 days WR
##stock['wr_10']
### 6 days WR
##stock['wr_6']
### CCI, default to 14 days
##stock['cci']
### 20 days CCI
##stock['cci_20']
### TR (true range)
##stock['tr']
### ATR (Average True Range)
##stock['atr']
### DMA, difference of 10 and 50 moving average
##stock['dma']
### DMI
### +DI, default to 14 days
##stock['pdi']
### -DI, default to 14 days
##stock['mdi']
### DX, default to 14 days of +DI and -DI
##stock['dx']
### ADX, 6 days SMA of DX, same as stock['dx_6_ema']
##stock['adx']
### ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
##stock['adxr']
##
### TRIX, default to 12 days
##stock['trix']
### MATRIX is the simple moving average of TRIX
##stock['trix_9_sma']

stock['close_28_ema']
stock['close_260_ema']

dt['diff']= dt['close_28_ema'] - dt['close_260_ema']
 
##plt.plot(dt.close_5_ema, label='EMA_5')
##plt.plot(dt.close_30_ema, label='EMA_30')
##plt.plot(dt.close, label='Price')     
#print(dt.tail(1))
#print(dt.columns)
#print(dt[['diff_5_21']])

#drop row if there is any value zero within a row
#final = dt.dropna(axis=0, how="any")
#print(final.head())

def fillWithMean(df):
  #return dt.fillna(dt.mean(), inplace=True)
  return df.apply(lambda x: x.fillna(x.mean()),axis=0) 

dt = fillWithMean(dt)
#final = dt.fillna(dt.mean())
print(dt.head())

# below is the logic when to start selling and when to buy
X = 0.05
dt['dec'] = np.where(dt['diff'] > X, 1, 0)
#dt['dec'] = np.where(dt['diff'] < X, -1, dt['dec'])
print("Total number of buy and Sell Signal \n",dt['dec'].value_counts())

# converting dataframe to dataseries array
dataset = dt.values

# split into input (X) and output (Y) variables
X = dataset[:,0:-1].astype(float)
Y = dataset[:,-1]

# splitting to train dataset and testdataset
X, tx, Y, ty = train_test_split(X, Y, test_size=0.2)

# making the netral network model 
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# compiling the model for binary output 
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# training the model for given data "X" features and "Y" binary output 			  
model.fit(X, Y,
          epochs=20,
          batch_size=100)

# evaluating the model 		  
score = model.evaluate(tx, ty, batch_size=100)
res = model.predict(tx)
# Creating the Confusion Matrix
#cm = confusion_matrix(ty, res)
plt.plot(res)
plt.show()
print("Score is: ",score,"\n",res,"\n")

