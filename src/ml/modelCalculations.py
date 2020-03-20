import sys
import os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import utilities
import plots
import metrics
import inspect
from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)
from models import Args
from models import (
    kerasSequentialRegressionModel,
    kerasSequentialRegressionModelWithRegularization,
    sklearnMLP,
    sklearnLinear,
    sklearnRidgeCV,
    ensembleModel,
    kerasLSTM,
    autoencoder_Dropout,
    autoencoder_Regularized,
)
from configs import (getConfig)

import matplotlib.pyplot as plt

args = {
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 500,
    'batchSize': 32,
    'verbose': 1,
    'callbacks': utilities.getBasicCallbacks(),
    'enrolWindow': 16,
    'validationSize': 0.2,
    'testSize': 0.2
}

lstmArgs = {
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 50,
    'batchSize': 32,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(),
    'enrolWindow': 1,
    'validationSize': 0.2,
    'testSize': 0.2
}

lstmArgs2 = {
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 50,
    'batchSize': 32,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(),
    'enrolWindow': 16,
    'validationSize': 0.2,
    'testSize': 0.2
}

def main(filename, targetColumns):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    traintime, testtime, validtime = timestamps

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    df_train, df_test = utilities.getTestTrainSplit(df, traintime, testtime)
    X_train, y_train, X_test, y_test = utilities.getFeatureTargetSplit(df_train, df_test, targetColumns)

    keras_seq_mod_regl = kerasSequentialRegressionModelWithRegularization(
        params = {
            'name': '50 20 regularized',
            'X_train': X_train,
            'y_train': y_train,
            'args': args,
        },
        structure = [
            [50, args['activation']],
            [20, args['activation']]
        ],
    )
    keras_seq_mod_simple = kerasSequentialRegressionModel(
        params = {
            'name': '20 normal',
            'X_train': X_train,
            'y_train': y_train,
            'args': args,
        },
        structure = [
            [20, args['activation']]
        ],
    )
    keras_seq_mod_v_simple = kerasSequentialRegressionModel(
        params = {
            'name': '0 Simple',
            'X_train': X_train,
            'y_train': y_train,
            'args': args,
        },
        structure = [
            [X_train.shape[1], args['activation']]
        ], 
    )
    keras_seq_mod = kerasSequentialRegressionModel(
        params = {
            'name': '50 20 normal',
            'X_train': X_train,
            'y_train': y_train,
            'args': args,
        },
        structure = [
            [50, args['activation']],
            [20, args['activation']]
        ]
    )
    lstmModel = kerasLSTM(
        params = {
            'name': 'LSTM 128',
            'X_train': X_train,
            'y_train': y_train,
            'args': lstmArgs,
        },
        units=[128],
        dropout=0.1,
        alpha=0.5
    )
    lstmModel2 = kerasLSTM(
        params = {
            'name': 'LSTM 2 128',
            'X_train': X_train,
            'y_train': y_train,
            'args': lstmArgs2,
        },
        units=[128],
        dropout=0.1,
        alpha=0.5
    )
    sklearnLinear = sklearnRidgeCV(
        params = {
            'name': 'Linear',
            'X_train': X_train,
            'y_train': y_train,
        },
    ) 
    ensemble = ensembleModel(
        params = {
            'name': 'ensemble 4 mods',
            'X_train': X_train,
            'y_train': y_train,
        },
        models = [
            keras_seq_mod_regl,
            keras_seq_mod_simple,
            lstmModel,
            sklearnLinear,
        ],
    )

    autoenc1 = autoencoder_Dropout(
        params = {
            'name':'autoencoder 1',
            'X_train': X_train,
            'args': args,
        },
    )

    autoenc2 = autoencoder_Regularized(
        params = {
            'name':'autoencoder 2',
            'X_train': X_train,
            'args': args,
        },
    )
    
    
    modelList = [
        autoenc1,
        autoenc2,
        #keras_seq_mod,
        #keras_seq_mod_simple,
        #ensemble,
        #lstmModel,
        #sklearnLinear,
    ]

    retrain = False
    maxEnrolWindow = utilities.findMaxEnrolWindow(modelList)

    utilities.trainModels(modelList, filename, ['all'], retrain)
    
    utilities.predictWithAutoencoderModels(modelList, df_test, X_test)

    #utilities.trainModels(modelList, filename, targetColumns, retrain)
    #names, r2_train, r2_test, deviationsList, columnsList = utilities.predictWithModels(modelList, X_train, y_train, X_test, y_test, targetColumns)
    #utilities.saveModels(modelList, filename, targetColumns)
    #utilities.printModelPredictions(names, r2_train, r2_test)
    #utilities.plotModelPredictions(plt, deviationsList, columnsList, df_test.iloc[maxEnrolWindow:].index, labelNames, traintime)

    """
    for model in modelList:
        printModelSummary(model)
    
    """
        

# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    targetCol = sys.argv[2:]
    main(filename, targetCol)
