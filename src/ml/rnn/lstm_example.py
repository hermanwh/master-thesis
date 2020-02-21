import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import importlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.layers import CuDNNLSTM
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

from configs import getConfig

import utilities

EPOCHS = 20
BATCH_SIZE = 16
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ENROL_WINDOW = 1

sc = MinMaxScaler(feature_range=(0,1))

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

columns, relevantColumns, columnDescriptions, columnUnits, timestamps = getConfig('B')
traintime, testtime, validtime = timestamps

def getLSTMModel(INPUT_DIM):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(INPUT_DIM,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


def lstm_128(x_shape, y_shape): 
    input_layer = Input(shape=(None,x_shape[-1]))
    layer_1 = layers.LSTM(128,
                         dropout = 0.3,
                         recurrent_dropout = 0.3,
                         return_sequences = True)(input_layer, training=True)

    output_layer = layers.Dense(y_shape[-1])(layer_1)
    
    model = Model(input_layer, output_layer) 
    return model

def lstm_test(batch_size, X):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

def getModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(50, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(20, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

def plotResults(column, df_test, df_train, pred_test, pred_train, y_test, y_train):
    fig, ax1 = plt.subplots()
    ax1.set_title("Test set values and predictions")
    color = 'darkgreen'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(column)
    ax1.plot(df_test.index[ENROL_WINDOW:], pred_test, color=color, label="Test predictions")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(1, axis='y')

    color = 'red'
    print(df_test.index.shape)
    print(df_test.index[ENROL_WINDOW:].shape)
    ax1.plot(df_test.index[ENROL_WINDOW:], y_test[ENROL_WINDOW:], color=color, label="Test targets", alpha=0.5)

    fig1, ax2 = plt.subplots()
    ax2.set_title("Training set values and predictions")
    color = 'darkgreen'
    ax2.set_xlabel('Date')
    ax2.set_ylabel(column)
    ax2.plot(df_train.index[ENROL_WINDOW:], pred_train, color=color, label="Train predictions")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(1, axis='y')

    color = 'red'
    ax2.plot(df_train.index[ENROL_WINDOW:], y_train[ENROL_WINDOW:], color=color, label="Train targets", alpha=0.5)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    fig1.autofmt_xdate()
    fig.autofmt_xdate()
    
    plt.show()

def train(df_train, df_test, column):
    print(f"Column: {column}")

    X_train = df_train.drop(column, axis=1).values
    y_train = df_train[column].values

    X_test = df_test.drop(column, axis=1).values
    y_test = df_test[column].values

    print(X_train)
    print(X_train.shape)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_train)
    print(X_train.shape)
    print(X_train.shape[0])

    train_generator = TimeseriesGenerator(X_train, y_train, length=ENROL_WINDOW, sampling_rate=1, batch_size=128)
    test_generator = TimeseriesGenerator(X_test, y_test, length=ENROL_WINDOW, sampling_rate=1, batch_size=128)
    
    train_X, train_y = train_generator[0]
    test_X, test_y = test_generator[0]

    train_samples = train_X.shape[0]*len(train_generator)
    test_samples = test_X.shape[0]*len(test_generator)

    print("Size of individual batches: {}".format(test_X.shape[1]))
    print("Number of total samples in training feature set: {}".format(train_samples))
    print("Number of samples in testing feature set: {}".format(test_samples))

    units = 128
    num_epoch = 300
    learning_rate = 0.00144

    model = Sequential()
    model.add(LSTM(units, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LeakyReLU(alpha=0.5)) 
    model.add(Dropout(0.1))
    model.add(Dense(1))

    adam = Adam(lr=learning_rate)
    # Stop training when a monitored quantity has stopped improving.
    callback = [EarlyStopping(monitor="loss", min_delta = 0.00001, patience = 15, mode = 'auto', restore_best_weights=True)] 

    
    # Using regression loss function 'Mean Standard Error' and validation metric 'Mean Absolute Error'
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    # fit network
    history = model.fit_generator(train_generator, \
                                    epochs=num_epoch, \
                                    validation_data=test_generator, \
                                    callbacks = callback, \
                                    verbose=2, \
                                    shuffle=False, \
                                    initial_epoch=0)
    
    
    utilities.printHorizontalLine()

    pred_train = model.predict(train_generator)
    pred_test = model.predict(test_generator)
    r2_train = r2_score(y_train[ENROL_WINDOW:], pred_train)
    r2_test = r2_score(y_test[ENROL_WINDOW:], pred_test)
    print('Dense R2 score train:', r2_train)
    print('Dense R2 score test:', r2_test)
    
    print("Plotting resulting graphs")
    plotResults(column, df_test, df_train, pred_test, pred_train, y_test, y_train)
    utilities.printHorizontalLine()
    

def main(fileName, column):
    utilities.printEmptyLine()
    
    print("Running", pyName)
    print("Learns model and predicts output values using the provided dataset")
    utilities.printHorizontalLine()

    df = pd.read_csv(fileName)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    start_train, end_train = traintime
    start_test, end_test = testtime

    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    df_test = utilities.getDataByTimeframe(df, start_test, end_test)

    train(df_train, df_test, column)

    utilities.printEmptyLine()

pyName = "training_lstm.py"
arguments = [
    "- file name (string)",
    "- target column (string)",
]

# usage: python ml/training.py datasets/file.csv PDT203
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        column = sys.argv[2]
    except:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()
    main(filename, column)
