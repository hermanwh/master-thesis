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
import modelFuncs
import plots
import prints
import analysis

import numpy as np
import tensorflow as tf
np.random.seed(100)
tf.random.set_seed(100)

default_MLP_args = {
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 700,
    'batchSize': 128,
    'verbose': 1,
    'callbacks': modelFuncs.getBasicCallbacks(),
    'enrolWindow': 0,
    'validationSize': 0.2,
    'testSize': 0.2,
}

default_LSTM_args = {
    'activation': 'tanh',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 300,
    'batchSize': 128,
    'verbose': 1,
    'callbacks': modelFuncs.getBasicCallbacks(),
    'enrolWindow': 32,
    'validationSize': 0.2,
    'testSize': 0.2,
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
        """
        FUNCTION:
            Used to initiate a pandas dataframe from file and provided metadata
        
        PARAMS:
            filename: str
                location of dataset file on disk in .csv format
            columns: List of list of column data
                Provided metadata of column names, column descriptions and column units
            irrelevantColumns: List of strings
                columnNames excluded from the dataset
        
        RETURNS:
            df: Pandas dataframe
                Dataframe generated from file and metadata
        """

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
        
        df = utilities.initDataframe(
            filename,
            relevantColumns,
            columnDescriptions,
        )

        self.df = df

        return df

    def getTestTrainSplit(self, traintime, testtime):
        """
        FUNCTION:
            Used to split training and testing rows into separate data frames
        
        PARAMS:
            traintime: List of list of string pairs
                start and end times indicating periods used for training
            testtime: List of string pair
                start and end time indicating period used for testing
                preferably the entire period of the dataset
        
        RETURNS:
            List[df_train, df_test]: [Pandas dataframe, Pandas dataframe]
                Dataframes of training and testing dataset rows
        """

        self.traintime = traintime
        self.testtime = testtime

        df_train, df_test = utilities.getTestTrainSplit(
            self.df,
            traintime,
            testtime,
        )

        self.df_train = df_train
        self.df_test = df_test

        return [df_train, df_test]

    def getFeatureTargetSplit(self, targetColumns):
        """
        FUNCTION:
            Used to split feature and target columns into separate arrays
        
        PARAMS:
            targetColumns: List of strings
                names of columns present in the dataset used as output(target) values
        
        RETURNS:
            List[X_train, y_train, X_test, y_test]: [Numpy array, Numpy array, Numpy array, Numpy array]
                Arrays of feature and target values for training and testing
        """

        self.targetColumns = targetColumns

        X_train, y_train, X_test, y_test =  utilities.getFeatureTargetSplit(
            self.df_train,
            self.df_test,
            targetColumns,
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return [X_train, y_train, X_test, y_test]

    def prepareDataframe(self, df, traintime, testtime, targetColumns):
        """
        FUNCTION:
            Combination of getTestTrainingSplit and getFeatureTargetSplit
            Used for even higher level programs where df_train and df_test are not needed
        
        PARAMS:
            df: Pandas dataframe
                dataframe generated from provided metadata
            traintime: List of list of string pairs
                start and end times indicating periods used for training
            testtime: List of string pair
                start and end time indicating period used for testing
                preferably the entire period of the dataset
            targetColumns: List of strings
                names of columns present in the dataset used as output(target) values
        
        RETURNS:
            List[X_train, y_train, X_test, y_test]: [Numpy array, Numpy array, Numpy array, Numpy array]
                Arrays of feature and target values for training and testing
        """

        df_train, df_test = getTestTrainSplit(
            df,
            traintime,
            testtime,
        )

        return getFeatureTargetSplit(
            df_train,
            df_test,
            targetColumns,
        )

    def initModels(self, modelList):
        """
        FUNCTION:
            Used to initiate the provided models by calculating required model parameters
        
        PARAMS:
            modelList: list of MachineLearningModel/EnsembleModel objects
                The models used to make predictions
        
        RETURNS:
            None
        """

        self.maxEnrolWindow = utilities.findMaxEnrolWindow(modelList)
        self.indexColumn = self.df_test.iloc[self.maxEnrolWindow:].index
        self.modelList = modelList

    def trainModels(self, retrain=False):
        """
        FUNCTION:
            Used to train the models previously provided in the initModels method
        
        PARAMS:
            retrain: boolean
                Indicates if the program should prefer to load existing models where possible
        
        RETURNS:
            None
        """

        modelFuncs.trainModels(
            self.modelList,
            self.filename,
            self.targetColumns,
            retrain
        )

    def predictWithModels(self, plot=True, interpol=False):
        """
        FUNCTION:
            Used to create a Neural Network model using multilayer perceptron
        
        PARAMS:
            plot: boolean
                Indicates if plots of the calculated predictions are desired
            interpol: boolean
                Indicates if interpolated functions for predictions should be plotted
        
        RETURNS:
            List[modelNames, metrics_train, metrics_test]: [list(Str), list(float), list(float)]
                Lists containing the names and train/test scores of the provided models
        """

        modelNames, metrics_train, metrics_test, deviationsList, columnsList = utilities.predictWithModels(
            self.modelList,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.targetColumns 
        )

        if plot:
            prints.printModelScores(
                modelNames,
                metrics_train,
                metrics_test
            )
            plots.plotModelPredictions(
                plt,
                deviationsList,
                columnsList,
                self.indexColumn,
                self.columnDescriptions,
                self.columnUnits,
                self.traintime,
                interpol=interpol,
            )
            plots.plotModelScores(
                plt,
                modelNames,
                metrics_train,
                metrics_test
            )

        return [modelNames, metrics_train, metrics_test, columnsList, deviationsList]

    def predictWithAutoencoderModels(self):
        """
        FUNCTION:
            Used to make predictions in the case where all
            models in self.modelList are of type AutoencoderModel
        """

        utilities.predictWithAutoencoderModels(
            self.modelList,
            self.df_test,
            self.X_test
        )

    def MLP(
            self,
            name,
            layers=[128],
            dropout=None,
            l1_rate=0.0,
            l2_rate=0.0,
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
        """
        FUNCTION:
            Used to create a Neural Network model using multilayer perceptron
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            layers: list of integers
                List of neuron size for each layer
            dropout: float
                Level of dropout regularization
            l1_rate: float
                Level of l1 (Lasso) regularization
            l2_rate: float
                Level of l2 (Ridge) regularization

        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        mlpLayers = []
        for layerSize in layers:
            mlpLayers.append([layerSize, activation])

        model = models.kerasMLP(
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
            dropout = dropout,
            l1_rate = l1_rate,
            l2_rate = l2_rate,
        )

        return model

    def LSTM(
        self,
        name,
        layers=[128],
        dropout=0.0,
        recurrentDropout=0.0,
        alpha=None,
        training=False,
        enrolWindow=default_LSTM_args['enrolWindow'],
        activation=default_LSTM_args['activation'],
        loss=default_LSTM_args['loss'],
        optimizer=default_LSTM_args['optimizer'],
        metrics=default_LSTM_args['metrics'],
        epochs=default_LSTM_args['epochs'],
        batchSize=default_LSTM_args['batchSize'],
        verbose=default_LSTM_args['verbose'],
        validationSize=default_LSTM_args['validationSize'],
        testSize=default_LSTM_args['testSize'],
        ):
        """
        FUNCTION:
            Used to create a Recurrent Neural Network model using
            Long-Short Term Memory neurons (LSTM). Uses 
            traditional dropout as regularization method
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            layers: list of integers
                List of neuron size for each layer
            dropout: float
                Level of dropout
            recurrentDropout: float
                Level of recurrent dropout
            alpha: float
                Alpha of the leaky relu function
            enrolWindow: int
                Number of samples used to make each prediction
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

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
            layers=layers,
            dropout=dropout,
            recurrentDropout=recurrentDropout,
            alpha=alpha,
            training=training,
        )
        
        return model

    def GRU(
        self,
        name,
        layers=[128],
        dropout=0.0,
        recurrentDropout=0.0,
        alpha=None,
        training=False,
        enrolWindow=default_LSTM_args['enrolWindow'],
        activation=default_LSTM_args['activation'],
        loss=default_LSTM_args['loss'],
        optimizer=default_LSTM_args['optimizer'],
        metrics=default_LSTM_args['metrics'],
        epochs=default_LSTM_args['epochs'],
        batchSize=default_LSTM_args['batchSize'],
        verbose=default_LSTM_args['verbose'],
        validationSize=default_LSTM_args['validationSize'],
        testSize=default_LSTM_args['testSize'],
        ):
        """
        FUNCTION:
            Used to create a Recurrent Neural Network model using
            Long-Short Term Memory neurons (LSTM). Uses 
            traditional dropout as regularization method
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            layers: list of integers
                List of neuron size for each layer
            dropout: float
                Level of dropout
            recurrentDropout: float
                Level of recurrent dropout
            alpha: float
                Alpha of the leaky relu function
            enrolWindow: int
                Number of samples used to make each prediction
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.kerasGRU(
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
            layers=layers,
            dropout=dropout,
            recurrentDropout=recurrentDropout,
            alpha=alpha,
            training=training,
        )
        
        return model

    def Linear(self, name):
        """
        FUNCTION:
            Used to create a Linear Machine Learning model
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.sklearnLinear(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
            },
        )

        return model

    def Linear_Regularized(self, name):
        """
        FUNCTION:
            Used to create a Linear Machine Learning model with built-in
            regularization and cross validation
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.sklearnRidgeCV(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
            },
        )

        return model

    def Ensemble(self, name, modelList):
        """
        FUNCTION:
            Used to create an Ensemble model, combining the prediction
            of n>1 machine learning methods using a linear regressor
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            modelList: list of MachineLearningModel objects
                A list of machine learning models used to construct the Ensemble model
        
        RETURNS:
            model: EnsembleModel
                Ensemble model object which behaves the same as any other MachineLearningModel
        """

        model = models.ensembleModel(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
            },
            models=modelList,
        )

        return model

    def Autoencoder_Regularized(
            self,
            name,
            l1_rate=10e-4,
            encodingDim=3,
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
        """
        FUNCTION:
            Used to create an Autoencoder model using multilayer perceptron
            and reguarlization by Lasso regluarization
            NB: Autoencoder models SHOULD NOT and CAN NOT
                be used together with other models, or
                as submodels to Ensemble models
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            l1_rate: float
                Level of L1 regularization
            encodingDim: int
                Size of autoencoder middle layer
        
        RETURNS:
            model: AutoencoderModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.autoencoder_Regularized(
            params = {
                'name': name,
                'X_train': self.X_train,
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
            l1_rate=l1_rate,
            encodingDim=encodingDim,
        )
        
        return model

    def Autoencoder_Dropout(
            self,
            name,
            dropout=0.0,
            encodingDim=3,
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
        """
        FUNCTION:
            Used to create an Autoencoder model using multilayer perceptron
            and reguarlization by Lasso regluarization
            NB: Autoencoder models SHOULD NOT and CAN NOT
                be used together with other models, or
                as submodels to Ensemble models
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            dropout: float
                Level of dropout
            encodingDim: int
                Size of autoencoder middle layer
        
        RETURNS:
            model: AutoencoderModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.autoencoder_Dropout(
            params = {
                'name': name,
                'X_train': self.X_train,
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
            dropout=dropout,
            encodingDim=encodingDim,
        )
        
        return model