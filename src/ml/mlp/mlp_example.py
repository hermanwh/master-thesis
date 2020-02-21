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

def getModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(50, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(20, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

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

    X_train = df_train.drop(targetColumn, axis=1).values
    y_train = df_train[targetColumn].values

    X_test = df_test.drop(targetColumn, axis=1).values
    y_test = df_test[targetColumn].values

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    scaler = MinMaxScaler(feature_range=(0,1))
    #scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = kerasSequentialRegressionModelWithRegularization([[50, ACTIVATION], [20, ACTIVATION]], X_train.shape[1])
    
    model = getModel(X_train.shape[1])
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.fit(X_train,
              y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=VERBOSE,
              callbacks=[
                callback, 
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor = 'loss', factor = 0.5, patience = 10, verbose = 1, min_lr=5e-4,
                )
              ]
            )
    

    #model = sklearnRidgeCV(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_metrics = utilities.compareMetrics(y_train, pred_train)
    test_metrics = utilities.compareMetrics(y_test, pred_test)

    print(train_metrics)
    print(test_metrics)
    
    pred_transpose = pred_train.reshape(-1, 1)
    y_transpose = y_train.reshape(-1, 1)
    y_test_transpose = y_test.reshape(-1, 1)
    y_train_transpose = y_train.reshape(-1, 1)

    utilities.plotDataColumn(df_train, plt, targetColumn, pred_train, y_train, labelNames)
    utilities.plotDataColumnSingle(df_train, plt, targetColumn, y_train_transpose - pred_train, labelNames)
    utilities.plotDataColumn(df_test, plt, targetColumn, pred_test, y_test, labelNames)
    utilities.plotDataColumnSingle(df_test, plt, targetColumn, y_test_transpose - pred_test, labelNames)
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv relevantColumns(bool)
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2]
    main(filename, targetCol)
