# external imports, keep as is
import sys
import os
import matplotlib.pyplot as plt

ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import src.utils.utilities as utilities
import src.utils.models as models

# specific imports, use as needed
from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)

# define dataset specifics

filename = "../HX-condition-monitoring/datasets/D/data.csv"

columns = [
    ['20TT001', 'Gas side inlet temperature', 'degrees'],
    ['20PT001', 'Gas side inlet pressure', 'barG'],
    ['20FT001', 'Gas side flow', 'M^3/s'],
    ['20TT002', 'Gas side outlet temperature', 'degrees'],
    ['20PDT001', 'Gas side pressure difference', 'bar'],
    ['50TT001', 'Cooling side inlet temperature', 'degrees'],
    ['50PT001', 'Cooling side inlet pressure', 'barG'],
    ['50FT001', 'Cooling side flow', 'M^3/s'],
    ['50TT002', 'Cooling side outlet temperature', 'degrees'],
    ['50PDT001', 'Cooling side pressure differential', 'bar'],
    ['50TV001', 'Cooling side valve opening', '%'],
]

irrelevantColumns = [
    '50FT001',
    '50PDT001',
    '20PDT001',
    '50TV001',
]

targetColumns = [
    '50TT002',
]

names = list(map(lambda el: el[0], columns))
descriptions = list(map(lambda el: el[1], columns))
units = list(map(lambda el: el[2], columns))

relevantColumns = list(filter(lambda col: col not in irrelevantColumns, map(lambda el: el[0], columns)))
columnUnits = dict(zip(names, units))
columnDescriptions = dict(zip(names, descriptions))

traintime = [
        ["2020-01-01 00:00:00", "2020-03-01 00:00:00"],
    ]

testtime = [
    "2020-01-01 00:00:00",
    "2020-07-01 00:00:00"
]

args = utilities.Args({
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
    'testSize': 0.2,
})

lstmArgs = utilities.Args({
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 50,
    'batchSize': 32,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(monitor="loss"),
    'enrolWindow': 1,
    'validationSize': 0.2,
    'testSize': 0.2
})

lstmArgs2 = utilities.Args({
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 50,
    'batchSize': 32,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(monitor="loss"),
    'enrolWindow': 16,
    'validationSize': 0.2,
    'testSize': 0.2
})


df = utilities.initDataframe(filename, relevantColumns, columnDescriptions)
df_train, df_test = utilities.getTestTrainSplit(df, traintime, testtime)
X_train, y_train, X_test, y_test = utilities.getFeatureTargetSplit(df_train, df_test, targetColumns)

keras_seq_mod_regl = models.kerasSequentialRegressionModelWithRegularization(
    params={
        'name': '50 20 regularized',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [50, args.activation],
        [20, args.activation]
    ],
)
keras_seq_mod_simple = models.kerasSequentialRegressionModel(
    params={
        'name': '20 normal',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [20, args.activation]
    ],
)
keras_seq_mod_v_simple = models.kerasSequentialRegressionModel(
    params={
        'name': '0 Simple',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [X_train.shape[1], args.activation]
    ], 
)
keras_seq_mod = models.kerasSequentialRegressionModel(
    params={
        'name': '50 20 normal',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [50, args.activation],
        [20, args.activation]
    ]
)
lstmModel = models.kerasLSTMSingleLayerLeaky(
    params={
        'name': 'LSTM 128',
        'X_train': X_train,
        'y_train': y_train,
        'args': lstmArgs,
    },
    units=128,
    dropout=0.1,
    alpha=0.5
)
lstmModel2 = models.kerasLSTMSingleLayerLeaky(
    params={
        'name': 'LSTM 2 128',
        'X_train': X_train,
        'y_train': y_train,
        'args': lstmArgs2,
    },
    units=128,
    dropout=0.1,
    alpha=0.5
)
sklearnLinearModel = models.sklearnRidgeCV(
    params={
        'name': 'Linear',
        'X_train': X_train,
        'y_train': y_train,
    },
) 
ensemble = models.ensembleModel(
    params={
        'name': 'ensemble 4 mods',
        'X_train': X_train,
        'y_train': y_train,
    },
    models=[
        keras_seq_mod_regl,
        keras_seq_mod_simple,
        lstmModel,
        sklearnLinearModel,
    ],
)


modelList = [
    keras_seq_mod,
    keras_seq_mod_simple,
    ensemble,
    lstmModel,
    sklearnLinearModel,
]

retrain = False
maxEnrolWindow = utilities.findMaxEnrolWindow(modelList)

utilities.trainModels(modelList, filename, targetColumns, retrain)
names, r2_train, r2_test, deviationsList, columnsList = utilities.predictWithModels(
    modelList,
    X_train,
    y_train,
    X_test,
    y_test,
    targetColumns
)
utilities.saveModels(modelList, filename, targetColumns)
#utilities.printModelPredictions(names, r2_train, r2_test)
utilities.plotModelPredictions(
    plt,
    deviationsList,
    columnsList,
    df_test.iloc[maxEnrolWindow:].index,
    columnDescriptions,
    traintime
)
