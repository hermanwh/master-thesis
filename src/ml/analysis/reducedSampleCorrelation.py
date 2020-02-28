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
from models import (
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

import matplotlib.pyplot as plt

def main(filename, targetColumn):
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

    cov = covmat(df_train)
    printCovMat(cov)

    relevantIndex = relevantColumns.index(targetColumn)

    covList = []
    for i, col in enumerate(relevantColumns):
        covList.append([col, cov[relevantIndex, i]])
    print(covList)

    sortedCovList = sorted(covList, key=lambda x: np.absolute(x[1]))
    sortedCovList = sortedCovList[1:]
    print(sortedCovList)

    y_train = df_train[targetColumn].values
    y_test = df_test[targetColumn].values

    r2_scores_test = []
    r2_scores_train = []

    for i in range(len(relevantColumns) - 1):
        X_train = df_train.copy()
        X_test = df_test.copy()
        for j in range(i):
            X_train = X_train.drop(sortedCovList[j][0], axis=1)
            X_test = X_test.drop(sortedCovList[j][0], axis=1)
        print("Columns:" + ",".join(X_train.columns))
        X_train = X_train.drop(targetColumn, axis=1).values
        X_test = X_test.drop(targetColumn, axis=1).values

        model = sklearnRidgeCV(X_train, y_train)
        model = model.train()

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_metrics = utilities.calculateMetrics(y_train, pred_train)
        test_metrics = utilities.calculateMetrics(y_test, pred_test)

        r2_scores_train.append(train_metrics[0])
        r2_scores_test.append(test_metrics[0])
        
        pred_transpose = pred_train.reshape(-1, 1)
        y_transpose = y_train.reshape(-1, 1)
        y_test_transpose = y_test.reshape(-1, 1)
        y_train_transpose = y_train.reshape(-1, 1)

        utilities.plotColumns(
            df_test,
            plt,
            [
                [
                    'Deviation', 
                    targetColumn,
                    y_test - pred_test,
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
                    targetColumn,
                    pred_test,
                    'darkgreen',
                    0.5,
                ],
                [
                    'Targets',
                    targetColumn,
                    y_test,
                    'red',
                    0.5,
                ]
            ],
            desc="Prediction vs. targets, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
        )
    print(r2_scores_train)
    print(r2_scores_test)
    plt.show()

    print('done')

# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2]
    main(filename, targetCol)
