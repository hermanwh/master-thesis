import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


import utilities
import inspect
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
from configs import (getConfig)
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

EPOCHS = 10000
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

    traintime, testtime, validtime = timestamps

    df = readDataFile(filename)
    df = getDataWithTimeIndex(df)
    df = df.dropna()

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    start_train, end_train = traintime
    start_valid, end_valid = validtime
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
    #scaler = MinMaxScaler(feature_range=(0,1))
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    keras_seq_mod_regl = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    keras_seq_mod_simple = kerasSequentialRegressionModel(X_train, y_train, [[20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    keras_seq_mod_v_simple = kerasSequentialRegressionModel(X_train, y_train, [[X_train.shape[1], ACTIVATION]],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    keras_seq_mod = kerasSequentialRegressionModel(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks])
    
    r1 = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks], l1_rate=1.0, l2_rate=1.0)
    r2 = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks], l1_rate=0.1, l2_rate=0.1)
    r3 = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks], l1_rate=0.01, l2_rate=0.01)
    r4 = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks], l1_rate=0.001, l2_rate=0.001)
    r5 = kerasSequentialRegressionModelWithRegularization(X_train, y_train, [[50, ACTIVATION], [20, ACTIVATION] ],  [LOSS, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VERBOSE, callbacks], l1_rate=0.0001, l2_rate=0.0001)
    

    modelsList = [
        #[keras_seq_mod, "MLP normal"],
        [keras_seq_mod_regl, "MLP regularized"],
        [keras_seq_mod_simple, "MLP simple"],
        #[keras_seq_mod_v_simple, "MLP very simple"],
        #[r1, "1.0 regulariation"],
        [r4, "0.001"],
        #[r5, "0.0001"],
        #[sklearnLinear(X_train, y_train), "linear"],
        [sklearnRidgeCV(X_train, y_train), "ridge"],
    ]

    names = []
    r2_train = []
    r2_test = []

    deviationsList = []
    columnsList = []
    for i in range(y_train.shape[1]):
        columnsList.append([])
        columnsList[i].append([
                        'Targets',
                        targetColumns[i],
                        y_test[:, i],
                        'red',
                        0.5,
                    ])

        deviationsList.append([])

    #colors = ['#92a8d1','#034f84','#f7cac9','#f7786b','#deeaee','#b1cbbb','#eea29a','#c94c4c']
    #colors = ['#686256','#c1502e','#587e76','#a96e5b','#454140','#bd5734','#7a3b2e']
    colors = [
        '#0C0910',
        '#453750',
        '#73648A',
        '#9882AC',
        '#A393BF',
        '#8AAA79',
        '#657153',
        '#837569',
        '#B7B6C2',
        '#D1D5DE',
        '#D58936',
        '#A44200',
        '#69140E',
        '#3C1518'
    ]

    for i, modObj in enumerate(modelsList):
        mod, name = modObj
        
        model = mod.train()

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_metrics = utilities.calculateMetrics(y_train, pred_train)
        test_metrics = utilities.calculateMetrics(y_test, pred_test)
        
        for j in range(y_train.shape[1]):
            columnsList[j].append([
                        name,
                        targetColumns[j],
                        pred_test[:, j],
                        colors[i],
                        0.3,
                    ])
            deviationsList[j].append([
                        name,
                        targetColumns[j],
                        y_test[:, j] - pred_test[:, j],
                        colors[i],
                        0.3,
                    ])
            """
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
            """

        r2_train.append(train_metrics[0])
        r2_test.append(test_metrics[0])
        names.append(name)        
    for i in range(y_train.shape[1]):
        utilities.plotColumns(
            df_test,
            plt,
            deviationsList[i],
            desc="Deviation, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
            interpol=True,
        )
        utilities.plotColumns(
            df_test,
            plt,
            columnsList[i],
            desc="Prediction vs. targets, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
            interpol=True,
        )

    y_pos = np.arange(len(names))

    plt.show()

    plt.ylabel('Stuff')
    plt.title('Things')

    plt.plot(names, r2_train)
    plt.plot(names, r2_test)

    """
    plt.bar(y_pos, r2_test, align='center', alpha=0.5)
    plt.bar(y_pos, r2_train, align='center', alpha=0.5)
    plt.xticks(y_pos, names)
    """

    plt.show()

    """

    #model = getModel(X_train.shape[1])
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.fit(X_train,
              y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=VERBOSE,
              )
    
    #model = sklearnRidgeCV(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_metrics = utilities.calculateMetrics(y_train, pred_train)
    test_metrics = utilities.calculateMetrics(y_test, pred_test)

    print(train_metrics)
    print(test_metrics)
    
    pred_transpose = pred_train.reshape(-1, 1)
    y_transpose = y_train.reshape(-1, 1)
    y_test_transpose = y_test.reshape(-1, 1)

    utilities.plotDataColumn(df_train, plt, column, pred_train, y_train, labelNames)
    utilities.plotDataColumnSingle(df_train, plt, column, y_transpose - pred_train, labelNames)
    utilities.plotDataColumn(df_test, plt, column, pred_test, y_test, labelNames)
    utilities.plotDataColumnSingle(df_test, plt, column, y_test_transpose - pred_test, labelNames)
    plt.show()

    """

# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
