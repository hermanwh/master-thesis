import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import utilities
import plots
import metrics
import tensorflow as tf
import numpy as np
from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)
from models import (kerasSequentialRegressionModel,
                    kerasSequentialRegressionModelWithRegularization,
                    sklearnMLP,
                    sklearnLinear,
                    sklearnRidgeCV
                    )
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs import (getConfig, getConfigDirs)
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
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

from models import (
    kerasLSTMSingleLayer,
    kerasLSTMSingleLayerLeaky,
    kerasLSTMMultiLayer,
)

ENROL_WINDOW = 1

def lstm(X_train, y_train, X_test, y_test):
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

    # Stop training when a monitored quantity has stopped improving.
    callbacks = [
        EarlyStopping(
            monitor="loss", min_delta = 0.00001, patience = 15, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = 'loss', factor = 0.5, patience = 10, verbose = 1, min_lr=5e-4,
        )
    ] 

    model = kerasLSTMSingleLayerLeaky(X_train, y_train, [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, ENROL_WINDOW], units=UNITS, dropout=0.1, alpha=0.5).model

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
    
    return [model, train_generator, test_generator]

def mlp(X_train, y_train, X_test, y_test):
    EPOCHS = 400
    BATCH_SIZE = 128
    TEST_SIZE = 0.2
    SHUFFLE = True
    VERBOSE = 1

    ACTIVATION = 'relu'
    LOSS = 'mean_squared_error'
    OPTIMIZER = 'adam'
    METRICS = ['mean_squared_error']

    scaler = MinMaxScaler(feature_range=(0,1))
    #scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    callbacks = [
        EarlyStopping(
            monitor="loss", min_delta = 0.00001, patience = 15, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = 'loss', factor = 0.5, patience = 10, verbose = 1, min_lr=5e-4,
        )
    ]

    model = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, None])
    #model = kerasSequentialRegressionModel(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    #model = sklearnRidgeCV(X_train, y_train)

    model.train()

    return [model.model, X_test, X_train]

def main(filename, targetColumns):
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

    [mlpModel, X_test_mlp, X_train_mlp] = mlp(X_train, y_train, X_test, y_test)
    [lstmModel, trainGen, testGen] = lstm(X_train, y_train, X_test, y_test)
    
    lstmPredictions = lstmModel.predict(trainGen)
    mlpPredictions = mlpModel.predict(X_train_mlp[ENROL_WINDOW:])

    predictions = np.concatenate((lstmPredictions, mlpPredictions), axis=1)

    linearModel = sklearnLinear(predictions, y_train[ENROL_WINDOW:])
    linearModel.train()

    lstm_test = lstmModel.predict(testGen)
    mlp_test = mlpModel.predict(X_test_mlp[ENROL_WINDOW:])

    train_metrics_lstm = metrics.calculateMetrics(y_train[ENROL_WINDOW:], lstmPredictions)
    test_metrics_lstm = metrics.calculateMetrics(y_test[ENROL_WINDOW:], lstm_test)

    print(train_metrics_lstm)
    print(test_metrics_lstm)


    train_metrics_mlp = metrics.calculateMetrics(y_train[ENROL_WINDOW:], mlpPredictions)
    test_metrics_mlp = metrics.calculateMetrics(y_test[ENROL_WINDOW:], mlp_test)

    print(train_metrics_mlp)
    print(test_metrics_mlp)


    asd = np.concatenate((lstm_test, mlp_test), axis=1)

    pred_train = linearModel.predict(predictions)
    pred_test = linearModel.predict(asd)

    train_metrics = metrics.calculateMetrics(y_train[ENROL_WINDOW:], pred_train)
    test_metrics = metrics.calculateMetrics(y_test[ENROL_WINDOW:], pred_test)

    print(train_metrics)
    print(test_metrics)

    for i in range(y_train.shape[1]):
        plots.plotColumns(
            df_test.iloc[ENROL_WINDOW:],
            plt,
            [
                [
                    'Prediction',
                    targetColumns[i],
                    lstm_test[:, i],
                    'darkgreen',
                    0.5,
                ],
                [
                    'Target',
                    targetColumns[i],
                    y_test[:, i][ENROL_WINDOW:],
                    'red',
                    0.5,
                ]
            ],
            desc="LSTM, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
        )
        plots.plotColumns(
            df_test.iloc[ENROL_WINDOW:],
            plt,
            [
                [
                    'Prediction',
                    targetColumns[i],
                    mlp_test[:, i],
                    'darkgreen',
                    0.5,
                ],
                [
                    'Target',
                    targetColumns[i],
                    y_test[:, i][ENROL_WINDOW:],
                    'red',
                    0.5,
                ]
            ],
            desc="MLP, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
        )
        plots.plotColumns(
            df_test.iloc[ENROL_WINDOW:],
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
                    y_test[:, i][ENROL_WINDOW:],
                    'red',
                    0.5,
                ]
            ],
            desc="Ensemble, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
        )

    plt.show()

# usage: python ml/covmat.py datasets/filename.csv targetCol
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
