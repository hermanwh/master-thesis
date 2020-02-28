import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import utilities
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
from utilities import (readDataFile,
                       getDataWithTimeIndex,
                       getDataByTimeframe,
                       printEmptyLine,
                       plotData,
                       plotDataColumnSingle
                       )
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs import (getConfig, getConfigDirs)
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

EPOCHS = 4000
BATCH_SIZE = 128
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

def main(filename, targetColumns):
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

    model = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    #model = kerasSequentialRegressionModel(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    #model = sklearnRidgeCV(X_train, y_train)

    model = model.train()

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_metrics = utilities.calculateMetrics(y_train, pred_train)
    test_metrics = utilities.calculateMetrics(y_test, pred_test)

    print(train_metrics)
    print(test_metrics)

    for i in range(y_train.shape[1]):
        utilities.plotColumns(
            df_test,
            plt,
            [
                [
                    'Deviation', 
                    targetColumns[i],
                    y_test[:, i] - pred_test[:, i],
                    'darkgreen',
                    0.5,
                ]
            ],
            desc="Deviation, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
        )
        utilities.plotColumns(
            df_test,
            plt,
            [
                [
                    'Predictions',
                    targetColumns[i],
                    pred_test[:, i],
                    'darkgreen',
                    0.5,
                ],
                [
                    'Targets',
                    targetColumns[i],
                    y_test[:, i],
                    'red',
                    0.5,
                ]
            ],
            desc="Prediction vs. targets, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
        )
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv targetCol
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
