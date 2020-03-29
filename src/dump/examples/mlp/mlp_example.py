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
import modelFuncs
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

from utilities import Args

args = Args({
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 4000,
    'batchSize': 128,
    'verbose': 1,
    'callbacks': modelFuncs.getBasicCallbacks(),
    'enrolWindow': None,
    'validationSize': 0.2,
    'testSize': 0.2
})

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

    #model = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, None])
    #model = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks, None], l1_rate=0.001, l2_rate=0.001)
    model = kerasSequentialRegressionModelWithRegularization(
        X_train,
        y_train,
        args,
        [
            [128, args.activation],
            [32, args.activation]
        ],
        l1_rate=0.001,
        l2_rate=0.001
    )
    #model = sklearnRidgeCV(X_train, y_train)

    model.train()

    plots.plotTraining(model.history, plt)

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
            trainEndStr=[end_train],
            interpol=False,
            interpoldeg=3,
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
            desc="Prediction targets",
            columnDescriptions=labelNames,
            columnUnits=columnUnits,
            trainEndStr=[end_train],
            interpol=False,
            interpoldeg=3
        )
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv targetCol
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
