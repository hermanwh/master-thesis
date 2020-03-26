import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import metrics
import plots
import prints

np.random.seed(100)
tf.random.set_seed(100)

def initDataframe(filename, relevantColumns, labelNames):
    df = readDataFile(filename)
    df = getDataWithTimeIndex(df)
    df = df.dropna()

    if relevantColumns is not None:
        df = dropIrrelevantColumns(df, [relevantColumns, labelNames])

    return df

def readDataFile(filename):
    ext = filename[-4:]
    if ext == '.csv':
        df = pd.read_csv(filename)
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        elif 'time' in df.columns:
            df['Date'] = df['time'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df.drop('time', axis=1)
    elif ext == '.xls':
        df = pd.read_excel(filename)
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        elif 'time' in df.columns:
            df['Date'] = df['time'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df.drop('time', axis=1)
    else:
        raise ValueError("Could not load data from file. Filename must be .csv or .xls format")
    return df

def getDataWithTimeIndex(df, dateColumn='Date'):
    if dateColumn in df.columns:
        df = df.set_index(dateColumn, inplace=False)
    else:
        raise ValueError('No date column named ' + dateColumn + '.')
    return df

def dropIrrelevantColumns(df, args):
    relevantColumns, columnDescriptions = args

    print("Columns before removal: ")
    prints.printColumns(df, columnDescriptions)

    dfcolumns = df.columns
    for column in dfcolumns:
        if column not in relevantColumns:
            df = df.drop(column, axis=1)

    prints.printEmptyLine()
    print("Columns after removal: ")
    prints.printColumns(df, columnDescriptions)
    prints.printEmptyLine()
    
    return df

def getTestTrainSplit(df, traintime, testtime):
    start_train, end_train = traintime[0]
    df_train = getDataByTimeframe(df, start_train, end_train)
    for start_train, end_train in traintime[1:]:
        nextDf = getDataByTimeframe(df, start_train, end_train)
        df_train = pd.concat([df_train, nextDf])

    if isinstance(testtime[0], str):
        start_test, end_test = testtime
        df_test = getDataByTimeframe(df, start_test, end_test)
    else:
        start_test, end_test = testtime[0]
        df_test = getDataByTimeframe(df, start_test, end_test)
        for start_test, end_test in testtime[1:]:
            nextDf = getDataByTimeframe(df, start_test, end_test)
            df_train = pd.concat([df_train, nextDf])

    return [df_train, df_test]

def getDataByTimeframe(df, start, end):
    print("Finding data between", start, "and", end)
    df = df.loc[start:end]
    print("Found " + str(df.shape[0]) + " rows")
    prints.printEmptyLine()
    return df

def getFeatureTargetSplit(df_train, df_test, targetColumns):
    X_train = df_train.drop(targetColumns, axis=1).values
    y_train = df_train[targetColumns].values

    X_test = df_test.drop(targetColumns, axis=1).values
    y_test = df_test[targetColumns].values

    return [X_train, y_train, X_test, y_test]

def predictWithModel(model, X_train, y_train, X_test, y_test, targetColumns):
    return predictWithModels([model], X_train, y_train, X_test, y_test, targetColumns)

def predictMultipleWithModels(modelList, X_test, y_test, numberOfPredictions):
    predictions = []
    means = []
    standarddevs = []

    for model in modelList:
        if model.modelType == "RNN":
            pred, mean, standarddev = model.predictMultiple(X_test, y_test, numberOfPredictions)
            predictions.append(pred)
            means.append(mean)
            standarddevs.append(standarddev)
    
    return [predictions, means, standarddevs]

def predictWithModels(modelList, X_train, y_train, X_test, y_test, targetColumns):
    colors = plots.getPlotColors()
    maxEnrol = findMaxEnrolWindow(modelList)

    names = []
    r2_train = []
    r2_test = []

    deviationsList = []
    columnsList = []
    for i in range(y_train.shape[1]):
        deviationsList.append([])
        columnsList.append([])
        columnsList[i].append(
            [
                'Targets',
                targetColumns[i],
                y_test[:, i][maxEnrol:],
                'red',
            ]
        )

    for i, modObj in enumerate(modelList):
        mod = modObj
        if mod.modelType == "Ensemble":
            enrol = mod.maxEnrol
        elif mod.modelType == "RNN":
            enrol = mod.args.enrolWindow
        else:
            enrol = 0
        enrolDiff = maxEnrol - enrol
        
        pred_train = mod.predict(X_train, y=y_train)
        pred_test = mod.predict(X_test, y=y_test)
        train_metrics = metrics.calculateMetrics(y_train[enrol:], pred_train)
        test_metrics = metrics.calculateMetrics(y_test[enrol:], pred_test)
        
        for j in range(y_train.shape[1]):
            columnsList[j].append(
                [
                    mod.name,
                    targetColumns[j],
                    pred_test[:, j][enrolDiff:],
                    colors[i],
                ]
            )
            deviationsList[j].append(
                [
                    mod.name,
                    targetColumns[j],
                    y_test[:, j][maxEnrol:] - pred_test[:, j][enrolDiff:],
                    colors[i],
                ]
            )

        r2_train.append(train_metrics[0])
        r2_test.append(test_metrics[0])
        names.append(mod.name)   
    
    return [
        names,
        r2_train,
        r2_test,
        deviationsList,
        columnsList,
    ]

def predictWithAutoencoderModels(modelList, df_test, X_test):
    indexx = df_test.index

    for modell in modelList:
        pred_test = modell.predict(X_test)

        for i in range(X_test.shape[1]):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
            ax.plot(indexx, pred_test[:, i], color='red', alpha=0.8)
            ax.plot(indexx, X_test[:, i], color='blue', alpha=0.8)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.set_ylabel('Value', fontsize=12)

            ax.set_title(df_test.columns[i], fontsize=16)

        plt.show()

        for i in range(X_test.shape[1]):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
            ax.plot(indexx, X_test[:, i] - pred_test[:, i], color='red', alpha=0.8)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.set_ylabel('Deviation', fontsize=12)

            ax.set_title(df_test.columns[i], fontsize=16)

        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        ax.plot(indexx, np.average((X_test - pred_test)**2,axis=1), color='red', alpha=0.8)
        ax.set_xlabel('Date', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_ylabel('Error', fontsize=12)

        ax.set_title('Reconstruction error', fontsize=16)

        plt.show()

def findMaxEnrolWindow(modelList):
    maxEnrol = 0
    for model in modelList:
        if model.modelType == "Ensemble":
            enrol = model.maxEnrol
        elif model.modelType == "RNN":
            enrol = model.args.enrolWindow
        else:
            enrol = 0

        if enrol > maxEnrol:
            maxEnrol = enrol

    return maxEnrol

def getColorScheme():
    return {
        'b1':"#0051FF",
        'b2':"#007CFF",
        'b3':"#4DA3FE",
        'r1':"#FF0101",
        'r2':"#FF3B3B",
        'r3':"#EB7D00",
        'r4':"#EBCF00",
    }

def testForGPU():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
