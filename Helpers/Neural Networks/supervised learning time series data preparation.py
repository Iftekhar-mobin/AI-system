from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
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
dataset = read_csv('GBPUSD240.csv', header=0, usecols=['High','Open','Close','Low'])

stock = Sdf.retype(dataset)
stock['close_21_ema']
#stock['boll']
stock['dma']
#stock['macds']

#print(dataset.describe())

values = dataset.values

#print(values)
### integer encode direction
encoder = LabelEncoder()
values[:,3] = encoder.fit_transform(values[:,3])
### ensure all data is float
values = values.astype('float32')

#print(values)

### normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

#print(scaled[0]) 

### frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

### drop columns we don't want to predict
reframed.drop(reframed.columns[[12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
##values = reframed.values
##n_train_hours = 365 * 24
##train = values[:n_train_hours, :]
##test = values[n_train_hours:, :]

# split into train and test sets
train_size = int(len(values) * 0.7)
test_size = len(values) - train_size
train, test = values[0:train_size,:], values[train_size:len(dataset),:]

# split into input and outputs
train_X, train_y = train[:, 0:8], train[:, -5:-1]
test_X, test_y = test[:, 0:8], test[:, -5:-1]

#print(train_y[0])

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

