from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from numpy import array
from math import sqrt
from numpy import concatenate, transpose, empty_like, nan
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from stockstats import StockDataFrame as Sdf

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('samples/GBPJPY240.csv', header=0, usecols=['High','Open','Close','Low'])

stock = Sdf.retype(dataset)
stock['close_21_ema']
#stock['boll']
stock['dma']
#stock['macds']

values = dataset.values
#print(values[:,0],values.shape)

print(dataset.describe())

new_val = values[:,0:1]
##print("New vaiable")
##print(new_val, new_val.shape)

sr = MinMaxScaler(feature_range=(0, 1))
sc = sr.fit_transform(new_val)
##print(sc, sc.shape)
##yhat = array([[ 0.20767153],[ 0.21361426],[ 0.22895732]])
##print (sr.inverse_transform(yhat))

# integer encode direction
encoder = LabelEncoder()
values[:,3] = encoder.fit_transform(values[:,3])
# ensure all data is float
values = values.astype('float32')
#print(transpose(values[:,0]),values.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
### specify the number of lag hours
n_hours = 4
n_features = 8
##
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[28,29,30,31]], axis=1, inplace=True)
#print(reframed.head())
## 
# split into train and test sets
values = reframed.values
train_size = int(len(values) * 1)
train = values[0:train_size,:]
##

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
#print(train_X.shape, len(train_X), train_y.shape)
##
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
# load model from single file
model = load_model('lstm_model_GBPJPY.h5')
# make predictions
yhat = model.predict(train_X, verbose=0)
##print ("yhat")
##print(yhat, yhat.shape)

inv_yhat = sr.inverse_transform(yhat)
print ("inv_yhat")
print(inv_yhat, inv_yhat.shape)

predict = yhat.ravel()
given_data = transpose(values[:,0]) 
print ("values column 0 transposed")
print(given_data, predict)

##mergedlist = []
##mergedlist.extend(t)
##mergedlist.extend(d)
##
##print (mergedlist)

#pyplot.plot(transpose(values[0:10,0]))


## calculate RMSE
rmse = sqrt(mean_squared_error(predict, given_data))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(given_data, label='data')
pyplot.plot(predict[n_hours:], label='predict')
pyplot.legend()
pyplot.show()
