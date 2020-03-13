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
from sklearn.model_selection import train_test_split

import pandas as pd
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

callbacks = [
        EarlyStopping(
            monitor="val_loss", min_delta = 0.00001, patience = 100, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = 'val_loss', factor = 0.5, patience = 100, verbose = 1, min_lr=5e-4,
        )
    ]

def main(filename, targetColumns):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    traintime, testtime, validtime = timestamps

    trainEndStrs = [
        "2016-01-01 00:00:00",
        "2016-03-01 00:00:00",
        "2016-05-01 00:00:00",
        "2016-06-01 00:00:00",
        "2016-08-01 00:00:00",
        "2016-10-01 00:00:00",
        "2017-07-01 00:00:00",
        "2017-08-01 00:00:00",
        "2017-10-01 00:00:00",
        "2017-11-01 00:00:00",
    ]

    traintime = [
        ["2016-01-01 00:00:00", "2016-03-01 00:00:00"],
        ["2016-05-01 00:00:00", "2016-06-01 00:00:00"],
        ["2016-08-01 00:00:00", "2016-10-01 00:00:00"],
        ["2017-07-01 00:00:00", "2017-08-01 00:00:00"],
        ["2017-10-01 00:00:00", "2017-11-01 00:00:00"],
    ]
	
    testtime = ["2016-01-01 00:00:00", "2020-03-01 00:00:00"]

    start_train, end_train = traintime[0]
    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    for start_train, end_train in traintime[1:]:
        nextDf = utilities.getDataByTimeframe(df, start_train, end_train)
        df_train = pd.concat([df_train, nextDf])

    start_test, end_test = testtime
    df_test = utilities.getDataByTimeframe(df, start_test, end_test)

    X_train = df_train.drop(targetColumns, axis=1).values
    y_train = df_train[targetColumns].values

    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_test = df_test.drop(targetColumns, axis=1).values
    y_test = df_test[targetColumns].values

    #scaler = MinMaxScaler(feature_range=(0,1))
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[128, ACTIVATION], [128, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, None], l1_rate=0.001, l2_rate=0.001)
    model.train()
    print(model.history)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_metrics = metrics.calculateMetrics(y_train, pred_train)
    test_metrics = metrics.calculateMetrics(y_test, pred_test)

    print(train_metrics)
    print(test_metrics)
    """
    with open(ROOT_PATH + "/src/ml/trained_models/" + subdir + "/metrics.txt", 'w') as output:
        output.write("Train metrics: " + str(train_metrics) + "\n")
        output.write("Test metrics: " + str(test_metrics) + "\n")
        output.write("Input columns: " + str(list(map((lambda x: labelNames[x]), list(df_train.drop(targetColumns, axis=1).columns)))) + "\n")
        output.write("Output columns: " + str(list(map((lambda x: labelNames[x]), list(df_train[targetColumns].columns)))) + "\n")
        
    model.save(ROOT_PATH + '/src/ml/trained_models/' + subdir + '/model.h5')
    """
    for i in range(y_train.shape[1]):
        plots.plotColumns(
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
            columnUnits=None,
            trainEndStr=trainEndStrs,
            interpol=False,
        )
        plots.plotColumns(
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
            columnUnits=columnUnits,
            trainEndStr=trainEndStrs,
            interpol=False,
        )
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv targetCol
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
