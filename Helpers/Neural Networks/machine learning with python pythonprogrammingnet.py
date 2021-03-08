import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

df = pd.read_csv("../python for finance/GBPAUD.csv", header=0, usecols=['Open', 'High', 'Low', 'Close', 'Volume'])

df = df[['Open',  'High',  'Low',  'Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

#print("forecasted value", forecast_out)
#print(df.head())

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#for k in ['linear','poly','rbf','sigmoid']  :
#    clf = svm.SVR(kernel=k)
#    clf.fit(X_train, y_train)
#    confidence = clf.score(X_test, y_test)
#    print(k,confidence)
    
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

#print("linearReg",confidence)

forecast_set = clf.predict(X_lately)

print(forecast_set, confidence, forecast_out)

close = df['Close'].values

nan_two = np.nan * np.ones(shape=(1,len(close)))
nan_one = nan_two.flatten()
total = np.concatenate((nan_one,forecast_set), axis=0)
#plt.plot(total[0:-len(forecast_set)])
plt.plot(total)
plt.plot(close)
plt.legend(loc=4)
plt.show()
