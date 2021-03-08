#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import numpy as np
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0:17]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0:4])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('GBPUSD240.csv',usecols=['High','Open','Close','Low'], engine='python', skipfooter=3)

##dataframe[['High','Open','Close','Low']].plot()
##plt.show()

#dataframe['5MA']  = dataframe['Close'].rolling(window=5).mean()
#dataframe['11MA']  = dataframe['Close'].rolling(window=11).mean()
#dataframe['EMA'] = dataframe['Close'].ewm(span=20).mean()
##
##dataframe[['5MA','11MA','EMA','High','Open','Close','Low']].plot()
##plt.show()

stock = Sdf.retype(dataframe)
#stock['close_100_ema']
#stock['rsi_14']
stock['boll']
stock['dma']
stock['macds']


##dataframe.plot()
##plt.show()

dataset = dataframe.values
dataset = dataset.astype('float32')
#split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#reshape into X=t and Y=t+1
look_back = 17
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#np.isnan(trainY)

print(trainX.shape[0])
print("--------------------------------------")
print(trainX.shape[1])
print("______________________________________")



