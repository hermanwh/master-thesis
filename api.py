import sys
import os

ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt

import utilities
import metrics
import models

from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)
from src.ml.analysis.pcaPlot import (pcaPlot, printReconstructionRow)

default_MLP_args = {
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 1000,
    'batchSize': 32,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(),
    'enrolWindow': 0,
    'validationSize': 0.2,
    'testSize': 0.2,
    'alpha': 0.5,
}

default_LSTM_args = {
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 100,
    'batchSize': 32*2,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(monitor="loss"),
    'enrolWindow': 16,
    'validationSize': 0.2,
    'testSize': 0.2,
    'dropout': 0.2,
    'recurrentDropout': 0.2,
    'alpha': 0.5,
}

class Api():
    def __init__(self):
        self.filename = None
        self.names = None
        self.descriptions = None
        self.units = None
        self.relevantColumns = None
        self.columnDescriptions = None
        self.columnUnits = None
        self.df = None
        self.traintime = None
        self.testtime = None
        self.df_train = None
        self.df_test = None
        self.targetColumns = None
        self.modelList = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.maxEnrolWindow = None
        self.indexColumn = None
    
    def initDataframe(self, filename, columns, irrelevantColumns):
        columnNames = list(map(lambda el: el[0], columns))
        descriptions = list(map(lambda el: el[1], columns))
        units = list(map(lambda el: el[2], columns))

        relevantColumns = list(filter(lambda col: col not in irrelevantColumns, map(lambda el: el[0], columns)))
        columnUnits = dict(zip(columnNames, units))
        columnDescriptions = dict(zip(columnNames, descriptions))

        self.filename = filename
        self.relevantColumns = relevantColumns
        self.columnDescriptions = columnDescriptions
        self.columnUnits = columnUnits
        self.columnNames = columnNames
        
        df = utilities.initDataframe(filename, relevantColumns, columnDescriptions)
        self.df = df
        return df

    def printMaxEnrol(self):
        print(self.maxEnrolWindow)

    def getTestTrainSplit(self, traintime, testtime):
        self.traintime = traintime
        self.testtime = testtime
        df_train, df_test = utilities.getTestTrainSplit(self.df, traintime, testtime)
        self.df_train = df_train
        self.df_test = df_test
        return [df_train, df_test]

    def getFeatureTargetSplit(self, targetColumns):
        self.targetColumns = targetColumns
        X_train, y_train, X_test, y_test =  utilities.getFeatureTargetSplit(self.df_train, self.df_test, targetColumns)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return [X_train, y_train, X_test, y_test]

    def prepareDataframe(self, df, traintime, testtime, targetColumns):
        df_train, df_test = getTestTrainSplit(df, traintime, testtime)
        return getFeatureTargetSplit(df_train, df_test, targetColumns)

    def initModels(self, modelList):
        self.maxEnrolWindow = utilities.findMaxEnrolWindow(modelList)
        self.indexColumn = self.df_test.iloc[self.maxEnrolWindow:].index
        self.modelList = modelList

    def trainModels(self, retrain=False):
        utilities.trainModels(self.modelList, self.filename, self.targetColumns, retrain)

    def predictWithModels(self, plot=True):
        modelNames, metrics_train, metrics_test, deviationsList, columnsList = utilities.predictWithModels(
            self.modelList,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.targetColumns 
        )
        if plot:
            utilities.plotModelPredictions(
                plt,
                deviationsList,
                columnsList,
                self.indexColumn,
                self.columnDescriptions,
                self.traintime
            )
            utilities.printModelPredictions(modelNames, metrics_train, metrics_test)
        return [modelNames, metrics_train, metrics_test]

    def MLP(
            self,
            name,
            layers=[128],
            activation=default_MLP_args['activation'],
            loss=default_MLP_args['loss'],
            optimizer=default_MLP_args['optimizer'],
            metrics=default_MLP_args['metrics'],
            epochs=default_MLP_args['epochs'],
            batchSize=default_MLP_args['batchSize'],
            verbose=default_MLP_args['verbose'],
            validationSize=default_MLP_args['validationSize'],
            testSize=default_MLP_args['testSize']
        ):

        mlpLayers = []
        for layerSize in layers:
            mlpLayers.append([layerSize, activation])

        model = models.kerasSequentialRegressionModel(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_MLP_args['callbacks'],
                    'enrolWindow': 0,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            structure = mlpLayers,
        )

        return model
    
    def MLP_Dropout(
            self,
            name,
            layers=[128],
            dropoutRate=0.2,
            activation=default_MLP_args['activation'],
            loss=default_MLP_args['loss'],
            optimizer=default_MLP_args['optimizer'],
            metrics=default_MLP_args['metrics'],
            epochs=default_MLP_args['epochs'],
            batchSize=default_MLP_args['batchSize'],
            verbose=default_MLP_args['verbose'],
            validationSize=default_MLP_args['validationSize'],
            testSize=default_MLP_args['testSize']
        ):

        mlpLayers = []
        for layerSize in layers:
            mlpLayers.append([layerSize, activation])

        model = models.kerasSequentialRegressionModelWithDropout(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_MLP_args['callbacks'],
                    'enrolWindow': 0,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            structure=mlpLayers,
            dropoutRate=dropoutRate
        )
        
        return model

    def MLP_Regularized(
            self,
            name,
            layers=[128],
            l1_rate=0.01,
            l2_rate=0.01,
            activation=default_MLP_args['activation'],
            loss=default_MLP_args['loss'],
            optimizer=default_MLP_args['optimizer'],
            metrics=default_MLP_args['metrics'],
            epochs=default_MLP_args['epochs'],
            batchSize=default_MLP_args['batchSize'],
            verbose=default_MLP_args['verbose'],
            validationSize=default_MLP_args['validationSize'],
            testSize=default_MLP_args['testSize']
        ):

        mlpLayers = []
        for layerSize in layers:
            mlpLayers.append([layerSize, activation])

        model = models.kerasSequentialRegressionModelWithRegularization(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_MLP_args['callbacks'],
                    'enrolWindow': 0,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            structure = mlpLayers,
            l1_rate=l1_rate,
            l2_rate=l2_rate,
        )
        
        return model

    def LSTM(
        self,
        name,
        units=[128],
        dropout=default_LSTM_args['dropout'],
        alpha=default_LSTM_args['alpha'],
        activation=default_LSTM_args['activation'],
        loss=default_LSTM_args['loss'],
        optimizer=default_LSTM_args['optimizer'],
        metrics=default_LSTM_args['metrics'],
        epochs=default_LSTM_args['epochs'],
        batchSize=default_LSTM_args['batchSize'],
        verbose=default_LSTM_args['verbose'],
        enrolWindow=default_LSTM_args['enrolWindow'],
        validationSize=default_LSTM_args['validationSize'],
        testSize=default_LSTM_args['testSize'],
        ):

        model = models.kerasLSTM(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_LSTM_args['callbacks'],
                    'enrolWindow': enrolWindow,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            units=units,
            dropout=dropout,
            alpha=alpha,
        )
        
        return model

    def LSTM_Recurrent(
        self,
        name,
        units=[128],
        dropout=default_LSTM_args['dropout'],
        recurrentDropout=default_LSTM_args['recurrentDropout'],
        alpha=default_LSTM_args['alpha'],
        activation=default_LSTM_args['activation'],
        loss=default_LSTM_args['loss'],
        optimizer=default_LSTM_args['optimizer'],
        metrics=default_LSTM_args['metrics'],
        epochs=default_LSTM_args['epochs'],
        batchSize=default_LSTM_args['batchSize'],
        verbose=default_LSTM_args['verbose'],
        enrolWindow=default_LSTM_args['enrolWindow'],
        validationSize=default_LSTM_args['validationSize'],
        testSize=default_LSTM_args['testSize'],
        ):

        model = models.kerasLSTM_Recurrent(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_LSTM_args['callbacks'],
                    'enrolWindow': enrolWindow,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            units=units,
            dropout=dropout,
            recurrentDropout=recurrentDropout,
        )
        
        return model

    def Linear(self, name):
        model = models.sklearnLinear(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
            },
        )

        return model

    def Linear_Regularized(self, name):
        model = models.sklearnRidgeCV(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
            },
        )

        return model

    def Ensemble(self, name, modelList):
        model = models.ensembleModel(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
            },
            models=modelList,
        )

        return model