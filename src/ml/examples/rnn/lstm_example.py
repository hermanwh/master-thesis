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

import utilities
import plots
import metrics

from models import (
    kerasLSTMSingleLayer,
    kerasLSTMSingleLayerLeaky,
    kerasLSTMMultiLayer,
)

from utilities import Args

args = Args({
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 1500,
    'batchSize': 128*2,
    'verbose': 2,
    'callbacks': utilities.getBasicCallbacks(monitor="loss"),
    'enrolWindow': 1,
    'validationSize': 0.2,
    'testSize': 0.2
})
    
def main(fileName, targetColumns):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
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

    model = kerasLSTMSingleLayerLeaky(
        X_train,
        y_train,
        args,
        units=128,
        dropout=0.1,
        alpha=0.5
    )
    model.train()

    """
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
    """
    
    utilities.printHorizontalLine()

    pred_train = model.predict(X_train, y=y_train)
    pred_test = model.predict(X_test, y=y_test)
    r2_train = metrics.r2_score(y_train[args.enrolWindow:], pred_train)
    r2_test = metrics.r2_score(y_test[args.enrolWindow:], pred_test)

    train_metrics = metrics.calculateMetrics(y_train[args.enrolWindow:], pred_train)
    test_metrics = metrics.calculateMetrics(y_test[args.enrolWindow:], pred_test)

    print(train_metrics)
    print(test_metrics)

    for i in range(y_train.shape[1]):
        plots.plotColumns(
            df_test.iloc[args.enrolWindow:],
            plt,
            [
                [
                    'Prediction',
                    targetColumns[i],
                    pred_test[:, i],
                    'darkgreen',
                    0.5,
                ],
                [
                    'Target',
                    targetColumns[i],
                    y_test[:, i][args.enrolWindow:],
                    'red',
                    0.5,
                ]
            ],
            desc="Prediction vs. targets, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
            interpol=True,
        )
        plots.plotColumns(
            df_test.iloc[args.enrolWindow:],
            plt,
            [
                [
                    'Deviation', 
                    targetColumns[i],
                    y_test[:, i][args.enrolWindow:] - pred_test[:, i],
                    'darkgreen',
                    0.5,
                ]
            ],
            desc="Deviation, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
            interpol=True,
        )

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
