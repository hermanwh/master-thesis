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
import matplotlib.pyplot as plt

from src.ml.analysis.covmat import (
    covmat,
    printCovMat,
)

from src.ml.analysis.pca import (
    pca,
    printExplainedVarianceRatio,
)

from models import (
    kerasSequentialRegressionModel,
    kerasSequentialRegressionModelWithRegularization,
    sklearnMLP,
    sklearnLinear,
    sklearnRidgeCV
)

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)

from configs import (
    getConfig,
    getConfigDirs,
)

from utilities import Args

params = Args({
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 4000,
    'batchSize': 128,
    'verbose': 1,
    'callbacks': utilities.getBasicCallbacks(),
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

    #scaler = MinMaxScaler(feature_range=(0,1))
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = kerasSequentialRegressionModel(
        X_train,
        y_train,
        [
            [128, params.activation],
            [32, params.activation]
        ],
        params)
    
    model.train()

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
            trainEndStr=[end_train],
            interpol=False,
        )
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv targetCol
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
