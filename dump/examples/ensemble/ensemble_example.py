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
    ensembleModel,
)

EPOCHS = 150
UNITS = 128
BATCH_SIZE = 128*2
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

ENROL_WINDOW = 1

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

    callbacks = [
        EarlyStopping(
            monitor="loss", min_delta = 0.00001, patience = 15, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = 'loss', factor = 0.5, patience = 10, verbose = 1, min_lr=5e-4,
        )
    ]

    mlpModel = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[128, ACTIVATION], [128, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, None], l1_rate=0.001, l2_rate=0.001)
    lstmModel = kerasLSTMSingleLayerLeaky(X_train, y_train, [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, ENROL_WINDOW], units=UNITS, dropout=0.1, alpha=0.5)
    linearModel = sklearnRidgeCV(X_train, y_train)

    maxEnrol = 1

    ensemble = ensembleModel([mlpModel, lstmModel, linearModel], X_train, y_train)
    ensemble.train()

    pred_train = ensemble.predict(X_train, y_train)
    pred_test = ensemble.predict(X_test, y_test)
    
    lstmPredictions = lstmModel.predict(X_train, y=y_train)
    mlpPredictions = mlpModel.predict(X_train)
    linearPredictions = linearModel.predict(X_train)

    lstm_test = lstmModel.predict(X_test, y=y_test)
    mlp_test = mlpModel.predict(X_test)
    linear_test = linearModel.predict(X_test)

    train_metrics_lstm = metrics.calculateMetrics(y_train[1:], lstmPredictions)
    test_metrics_lstm = metrics.calculateMetrics(y_test[1:], lstm_test)

    print(train_metrics_lstm)
    print(test_metrics_lstm)

    train_metrics_mlp = metrics.calculateMetrics(y_train, mlpPredictions)
    test_metrics_mlp = metrics.calculateMetrics(y_test, mlp_test)

    print(train_metrics_mlp)
    print(test_metrics_mlp)

    train_metrics_linear = metrics.calculateMetrics(y_train, linearPredictions)
    test_metrics_linear = metrics.calculateMetrics(y_test, linear_test)

    print(train_metrics_linear)
    print(test_metrics_linear)

    train_metrics = metrics.calculateMetrics(y_train[maxEnrol:], pred_train)
    test_metrics = metrics.calculateMetrics(y_test[maxEnrol:], pred_test)

    print(train_metrics)
    print(test_metrics)

    for i in range(y_train.shape[1]):
        plots.plotColumns(
            df_test.iloc[1:],
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
                    y_test[:, i][1:],
                    'red',
                    0.5,
                ]
            ],
            desc="LSTM, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
        )
        plots.plotColumns(
            df_test,
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
                    y_test[:, i],
                    'red',
                    0.5,
                ]
            ],
            desc="MLP, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
        )
        plots.plotColumns(
            df_test,
            plt,
            [
                [
                    'Prediction',
                    targetColumns[i],
                    linear_test[:, i],
                    'darkgreen',
                    0.5,
                ],
                [
                    'Target',
                    targetColumns[i],
                    y_test[:, i],
                    'red',
                    0.5,
                ]
            ],
            desc="Linear, ",
            columnDescriptions=labelNames,
            trainEndStr=[end_train],
        )
        plots.plotColumns(
            df_test.iloc[maxEnrol:],
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
                    y_test[:, i][maxEnrol:],
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
