import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, Imputer
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import load_model
from tqdm import trange

def make_forecast(model: Sequential, look_back_buffer: np.ndarray, timesteps: int=1, batch_size: int=1):
    #forecast_predict = np.empty((0, 1,len(look_back_buffer)), dtype=np.float32)
    forecast_predict = np.empty((1, 4), dtype=np.float32)
    forecast_predict[:, :] = np.nan
    #print("forecast_predict and lookBbuffer", forecast_predict.shape,look_back_buffer.shape)
    
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        #cur_predict = model.predict(look_back_buffer, batch_size)
        cur_predict = model.predict(look_back_buffer)
        #print("Current predict", cur_predict, cur_predict.shape)

        # add prediction to result
        forecast_predict = np.concatenate((forecast_predict, cur_predict))
        #print("forecast predict", forecast_predict, forecast_predict.shape)

        # deleted the oldest data from the input array
        look_back_buffer = np.delete(look_back_buffer, 0, axis=1)
        #print("look_back_buffer",look_back_buffer,look_back_buffer.shape)
        
        # reshapping the predicted output for concatenate with the input data                   
        cur_predict = cur_predict.reshape(1,1,4)
        #print("Current predict after reshape", cur_predict, cur_predict.shape)
		 
        #look_back_buffer = np.dstack((look_back_buffer, cur_predict))
        #look_back_buffer = np.append(look_back_buffer, cur_predict, axis=0)
        look_back_buffer = np.concatenate((look_back_buffer, cur_predict), axis=1)
        #print("look_back_buffer after adding",look_back_buffer,look_back_buffer.shape)
                
    return forecast_predict

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = pd.read_csv('samples/GBPJPY240.csv', header=0, usecols=['High','Open','Close','Low'])

#dataset.drop('dma', axis=1, inplace=True)

#print("column number")
#print(dataset.columns,len(dataset.columns),len(dataset.index))

dt = dataset.values
d = dt.astype(float)

#print("Checkinf for NaN and Inf")
#print( "np.nan=", np.where(np.isnan(d)))
#print( "is.inf=", np.where(np.isinf(d)))
#
print("********************************************")
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(d)
d = imp.fit_transform(d)

##print("values after encoding", values)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(d)
##print("scaled values",scaled)
# specify the number of lag hours
n_hours = 4
n_features = len(dataset.columns)
n_ahead = 1
st = n_hours*n_features
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, n_ahead)
#print("column number")
#print(reframed.columns,len(reframed.columns))
#
## drop columns we don't want to predict
##deletedcol = list(reframed.columns)[st+n_ahead : -1] 

#deletedcol = (
#        list(reframed.columns)[st : -n_hours] 
#        ) 
#reframed.drop(deletedcol, axis=1, inplace=True)
#
#deletedcol = (
#        list(reframed.columns)[n_hours : st]
#        ) 
# 
#reframed.drop(deletedcol, axis=1, inplace=True)
####print("deleted column",deletedcol)
###reframed.drop(reframed.columns[[25,26,27,28,29,30,31, 33,34,35,36,37,38,39,41,42,43,44,45,46,47]], axis=1, inplace=True)
#
##print("remaining column number")
##print(reframed.columns,len(reframed.columns))
##print(reframed.head())
#
## split into train and test sets
values = reframed.values
train_size = int(len(values) * 0.8)
test_size = len(values) - train_size
train, test = values[0:train_size,:], values[train_size:len(dataset),:]
#
## split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, 0:-4], train[:, -4:]
test_X, test_y = test[:, 0:-4], test[:, -4:]
#print(train_X, train_X.shape, train_y, train_y.shape)
#
## reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 4, n_hours))
test_X = test_X.reshape((test_X.shape[0], 4, n_hours))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
##
batchsize = 100
##
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')
##
## fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=batchsize, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#
## list all data in history
print(history.history.keys())
#
### save model to single file
#model.save('lstm_model_mutivariate_multistep_multiout.h5')
## 
## summarize history for accuracy
##pyplot.plot(history.history['acc'])
##pyplot.plot(history.history['val_acc'])
##pyplot.title('model accuracy')
##pyplot.ylabel('accuracy')
##pyplot.xlabel('epoch')
##pyplot.legend(['train', 'test'], loc='upper left')
##pyplot.show()
#   
## plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
##
## make a prediction
trainPredict = model.predict(train_X)
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_obs))
train_X = train_X.reshape((train_X.shape[0], n_obs))
#
print("yhat",yhat,yhat.shape,"test_X",test_X.shape,train_X.shape)

pyplot.plot(test_X[:,-4:])
pyplot.plot(yhat)
#pyplot.show()
#
## generate forecast predictions
test_X = test_X.reshape((test_X.shape[0], 4, n_hours))
print("testing forecast",test_X[-1::], test_X.shape)

batchsize = 20
forecast_step = 10

forecast_predict = make_forecast(model, test_X[-1::], timesteps=forecast_step, batch_size=batchsize)
print("forecast_predict",forecast_predict,forecast_predict.shape)
#
### shift test predictions for plotting
forecastPredictPlot = np.empty_like(yhat)
forecastPredictPlot[:, :] = np.nan
forecastplot = np.concatenate((forecastPredictPlot,forecast_predict))
#
pyplot.plot(forecastplot)
pyplot.show()
#
##nan_two = np.nan * np.ones(shape=(1,len(train_X[:,0])+len(test_X[:,0])))
##nan_one = nan_two.flatten()
##p1 = np.concatenate((nan_one,forecast_predict[:,n_ahead-n_ahead]), axis=0)
