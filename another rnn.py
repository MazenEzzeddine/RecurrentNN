
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed


def train_test_plot(dataset, tstart, tend):
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{tend + 1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend + 1})", f"Test ({tend + 1} and beyond)"])
    plt.title("MasterCard stock price")
    plt.show()




def train_test_split(dataset, tstart, tend):
    train = dataset.loc[f"{tstart}":f"{tend}", "High"].values
    test = dataset.loc[f"{tend+1}":, "High"].values
    return train, test

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("MasterCard Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("MasterCard Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))




if __name__ == '__main__':


    # loadDataset()
    # train_test_plot()
    set_seed(455)
    np.random.seed(455)
    dataset = pd.read_csv(
        "Mastercard_stock_history.csv", index_col="Date", parse_dates=["Date"]
    ).drop(["Dividends", "Stock Splits"], axis=1)
    print(dataset.head())

    tstart = 2016
    tend = 2020
    train_test_plot(dataset, tstart, tend)

    training_set, test_set = train_test_split(dataset, tstart, tend)

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = training_set.reshape(-1, 1)
    training_set_scaled = sc.fit_transform(training_set)

    n_steps = 60
    features = 1
    # split into samples
    X_train, y_train = split_sequence(training_set_scaled, n_steps)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

    model_lstm = Sequential()
    model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
    model_lstm.add(Dense(units=1))
    # Compiling the model
    model_lstm.compile(optimizer="RMSprop", loss="mse")

    model_lstm.summary()

    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)

    dataset_total = dataset.loc[:, "High"]
    inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values
    inputs = inputs.reshape(-1, 1)
    # scaling
    inputs = sc.transform(inputs)

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)
    # reshape
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)
    # prediction
    predicted_stock_price = model_lstm.predict(X_test)
    # inverse transform the values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    plot_predictions(test_set, predicted_stock_price)

    return_rmse(test_set, predicted_stock_price)














