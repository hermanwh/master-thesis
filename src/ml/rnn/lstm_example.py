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

from configs import (getConfig, getConfigDirs)

from utilities import (readDataFile,
                       getDataWithTimeIndex,
                       getDataByTimeframe,
                       printEmptyLine,
                       plotData,
                       plotDataColumnSingle
                       )

import utilities

EPOCHS = 300
UNITS = 128
BATCH_SIZE = 128
TEST_SIZE = 0.2
SHUFFLE = False
VERBOSE = 2
LEARNING_RATE = 0.00144

ENROL_WINDOW = 1

sc = MinMaxScaler(feature_range=(0,1))

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

def getModel(train_X, y_train, ALPHA=0.5, DROPOUT=0.1):
    model = Sequential()
    model.add(LSTM(UNITS, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LeakyReLU(alpha=ALPHA)) 
    model.add(Dropout(DROPOUT))
    model.add(Dense(y_train.shape[1]))
    return model

def getLSTMModel(inputDim, outputDim=1, dropoutRate=0.2):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(inputDim,1)))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(dropoutRate))
    model.add(Dense(outputDim))
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
    
def main(fileName, targetColumns):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = readDataFile(filename)
    df = getDataWithTimeIndex(df)
    df = df.dropna()

    traintime, testtime, validtime = timestamps

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    start_train, end_train = traintime
    start_test, end_test = testtime

    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    df_test = utilities.getDataByTimeframe(df, start_test, end_test)

    X_train = df_train.drop(targetColumns, axis=1).values
    y_train = df_train[targetColumns].values

    X_test = df_test.drop(targetColumns, axis=1).values
    y_test = df_test[targetColumns].values

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_generator = TimeseriesGenerator(X_train, y_train, length=ENROL_WINDOW, sampling_rate=1, batch_size=BATCH_SIZE)
    test_generator = TimeseriesGenerator(X_test, y_test, length=ENROL_WINDOW, sampling_rate=1, batch_size=BATCH_SIZE)

    train_X, train_y = train_generator[0]
    test_X, test_y = test_generator[0]

    train_samples = train_X.shape[0]*len(train_generator)
    test_samples = test_X.shape[0]*len(test_generator)

    print("Size of individual batches: {}".format(test_X.shape[1]))
    print("Number of total samples in training feature set: {}".format(train_samples))
    print("Number of samples in testing feature set: {}".format(test_samples))

    # Stop training when a monitored quantity has stopped improving.
    callbacks = [
        EarlyStopping(
            monitor="loss", min_delta = 0.00001, patience = 15, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = 'loss', factor = 0.5, patience = 10, verbose = 1, min_lr=5e-4,
        )
    ] 

    model = getModel(train_X, y_train)

    # Using regression loss function 'Mean Standard Error' and validation metric 'Mean Absolute Error'
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

    # fit network
    history = model.fit_generator(train_generator, \
                                    epochs=EPOCHS, \
                                    validation_data=test_generator, \
                                    callbacks = callbacks, \
                                    verbose=VERBOSE, \
                                    shuffle=SHUFFLE, \
                                    initial_epoch=0)
    
    utilities.printHorizontalLine()

    pred_train = model.predict(train_generator)
    pred_test = model.predict(test_generator)
    r2_train = r2_score(y_train[ENROL_WINDOW:], pred_train)
    r2_test = r2_score(y_test[ENROL_WINDOW:], pred_test)

    train_metrics = utilities.compareMetrics(y_train[ENROL_WINDOW:], pred_train)
    test_metrics = utilities.compareMetrics(y_test[ENROL_WINDOW:], pred_test)

    print(train_metrics)
    print(test_metrics)

    print(y_train.shape)
    print(pred_train.shape)

    for i in range(y_train.shape[1]):
        print(y_train[:, i])
        utilities.plotDataColumn(df_train.iloc[ENROL_WINDOW:], plt, targetColumns[i], pred_train[:, i], y_train[:, i][ENROL_WINDOW:], labelNames)
        utilities.plotDataColumnSingle(df_train.iloc[ENROL_WINDOW:], plt, targetColumns[i], y_train[:, i][ENROL_WINDOW:] - pred_train[:, i], labelNames)
        utilities.plotDataColumn(df_test.iloc[ENROL_WINDOW:], plt, targetColumns[i], pred_test[:, i], y_test[:, i][ENROL_WINDOW:], labelNames)
        utilities.plotDataColumnSingle(df_test.iloc[ENROL_WINDOW:], plt, targetColumns[i], y_test[:, i][ENROL_WINDOW:] - pred_test[:, i], labelNames)
    plt.show()


pyName = "training_lstm.py"
arguments = [
    "- file name (string)",
    "- target columns (sequence of strings)",
]

# usage: python ml/training.py datasets/file.csv col1 col2 col3 ...
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        targetColumns = sys.argv[2:]
    except:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()
    main(filename, targetColumns)
