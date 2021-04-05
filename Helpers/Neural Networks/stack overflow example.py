n_test = int(df.shape[0] * 0.2)
n_in = 300
n_out = 60
n_batch = 1
n_epochs = 3
verbose = 1
neurons = (5,5)

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

def prepare_data(series, n_test, n_lag, n_seq):
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(raw_values)
    # series_to_supervised is above
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

scaler, train, test = prepare_data(df, n_test, n_in, n_out)

def fit_lstm(train, test, n_lag, n_seq, n_batch, nb_epoch, n_neurons, verbose):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, :n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X_test, y_test = test[:, :n_lag], test[:, n_lag:]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    model = Sequential()
    model.add(LSTM(n_neurons[0], batch_input_shape=(n_batch, X.shape[1], X.shape[2]), 
                   return_sequences=True, stateful=True, dropout=0.4))
    model.add(LSTM(n_neurons[1], batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    losses = []
    val_losses = []
    min_val_loss = (99999,999999)
    for i in range(nb_epoch):
        if verbose!=0:
            print(i)
        history = model.fit(X, y, validation_data=(X_test,y_test), epochs=1, batch_size=n_batch, verbose=verbose, shuffle=False)
        losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'][0])
        if val_losses[-1] < min_val_loss[0]:
            min_val_loss = (val_losses[-1], i)
        model.reset_states()
    print('best val_loss and epoch:',min_val_loss)
    plt.title('loss')
    plt.plot(losses)
    plt.plot(val_losses, color='red')
    plt.show()
    return model

model = fit_lstm(train, test, n_in, n_out, n_batch, n_epochs, neurons, verbose)

def forecast_lstm(model, X, n_batch):
    X = X.reshape(1, 1, len(X))
    model.reset_states()
    forecast = model.predict(X, batch_size=n_batch, verbose=0)
    model.reset_states()
    return [x for x in forecast[0, :]]

def make_forecasts(model, n_batch, points, n_lag, n_seq):
    forecasts = list()
    for i in range(len(points)):
        X = points[i, 0:n_lag]
        forecast = forecast_lstm(model, X, n_batch)
        forecasts.append(forecast)
    return forecasts

forecasts = make_forecasts(model, n_batch, test, n_in, n_out)

def inverse_transform(forecasts, scaler):
    inverted = list()
    for i in range(len(forecasts)):
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        inverted.append(list(inv_scale))
    return inverted

forecasts = inverse_transform(forecasts, scaler)
actual = inverse_transform(test, scaler)

def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    total_rmse = 0
    if type(test) is list:
        test = np.array(test)
    for i in range(n_seq):
        actual = test[:,(n_lag+i)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
        total_rmse += rmse
    print('total rmse: ', total_rmse)

evaluate_forecasts(actual,forecasts,n_in, n_out)

def plot_forecasts(series, test, forecasts, n_in, n_out):
    t = pd.DataFrame(test)
    f = pd.DataFrame(forecasts)
    t.iloc[:,n_in:n_in+n_out] = f.values
    t['idx'] = len(series) + t.index.values - n_in - len(test) -n_out
    # plot the forecasts in red
    for i in range(len(forecasts)):
        xaxis = np.array([t.loc[i,'idx']] * (n_in+n_out)) + np.array(range((n_in+n_out)))
        yaxis = t.iloc[i,:-1].values
        plt.plot(xaxis, yaxis, color='red')
    # plot the entire dataset in blue
    plt.plot(series.values)
    plt.show()

plot_forecasts(train,actual,forecasts[-1],n_in,n_out)