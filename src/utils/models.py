from sklearn.linear_model import (ElasticNet, ElasticNetCV, LinearRegression, Lasso, LassoCV, Ridge, RidgeCV)
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
from copy import deepcopy

import os

class EnsembleModel():
    def __init__(self, models, X_train, y_train, modelType="Ensemble", name=None):
        maxEnrol = 0
        for model in models:
            if model.args is not None:
                enrol = model.args.enrolWindow
                if enrol is not None and enrol > maxEnrol:
                    maxEnrol = enrol

        self.maxEnrol = maxEnrol
        self.models = models
        self.MLmodel = None
        self.X_train = X_train
        self.y_train = y_train
        self.name = name
        self.history = None
        self.modelType = modelType

    def train(self):
        preds = []
        for model in self.models:
            model.train()
            prediction = model.predict(model.X_train, model.y_train)
            if model.modelType == "RNN":
                preds.append(prediction[self.maxEnrol - model.args.enrolWindow:])
            else:
                preds.append(prediction[self.maxEnrol:])

        train = preds[0]
        for pred in preds[1:]:
            train = np.concatenate((train, pred), axis=1)
        self.MLmodel = sklearnLinear(
            params = {
                'name': 'ML model of ensemble',
                'X_train': train,
                'y_train': self.y_train[self.maxEnrol:],
            },
        )
        self.MLmodel.train()

    def trainEnsemble(self):
        preds = []
        for model in self.models:
            prediction = model.predict(model.X_train, model.y_train)
            if model.modelType == "RNN":
                preds.append(prediction[self.maxEnrol - model.args.enrolWindow:])
            else:
                preds.append(prediction[self.maxEnrol:])

        train = preds[0]
        for pred in preds[1:]:
            train = np.concatenate((train, pred), axis=1)
        self.MLmodel = sklearnLinear(
            params = {
                'name': 'ML model of ensemble',
                'X_train': train,
                'y_train': self.y_train[self.maxEnrol:],
            },
        )
        self.MLmodel.train()

    def predict(self, X, y):
        preds = []
        for model in self.models:
            prediction = model.predict(X, y)
            if model.modelType == "RNN":
                preds.append(prediction[self.maxEnrol - model.args.enrolWindow:])
            else:
                preds.append(prediction[self.maxEnrol:])

        test = preds[0]
        for pred in preds[1:]:
            test = np.concatenate((test, pred), axis=1)
        return self.MLmodel.predict(test)

    def save(self, directory, name):
        for model in self.models:
            if model.args:
                dirr = directory + name + '/'
                if not os.path.exists(dirr):
                    os.makedirs(dirr)
                model.save(dirr, "_".join(model.name.split(' ')))

class MachinLearningModel():
    def __init__(self, model, X_train, y_train, args=None, modelType=None, scaler="standard", name=None):
        if scaler == "standard":
            inputScaler = StandardScaler()
            outputScaler = StandardScaler()
        else:
            inputScaler = MinMaxScaler()
            outputScaler = MinMaxScaler()
        
        inputScaler.fit(X_train)
        outputScaler.fit(y_train)

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.args = args
        self.name = name
        self.history = None
        self.inputScaler = inputScaler
        self.outputScaler = outputScaler
        self.modelType = modelType

    def train(self):
        if self.modelType == "RNN":
            train_generator = TimeseriesGenerator(
                self.inputScaler.transform(self.X_train),
                self.outputScaler.transform(self.y_train),
                length = self.args.enrolWindow,
                sampling_rate = 1,
                batch_size = self.args.batchSize
            )
            self.model.compile(
                loss = self.args.loss,
                optimizer = self.args.optimizer,
                metrics = self.args.metrics
            )
            self.history = self.model.fit_generator(
                train_generator,
                epochs = self.args.epochs,
                verbose = self.args.verbose,
                callbacks = self.args.callbacks,
            )
        elif self.modelType == "MLP":
            self.model.compile(
                loss = self.args.loss,
                optimizer = self.args.optimizer,
                metrics = self.args.metrics
            )
            self.history = self.model.fit(
                self.inputScaler.transform(self.X_train),
                self.outputScaler.transform(self.y_train),
                epochs = self.args.epochs,
                batch_size = self.args.batchSize,
                verbose = self.args.verbose,
                callbacks = self.args.callbacks,
                validation_split = self.args.validationSize,
            )
        else:
            self.history = self.model.fit(
                self.inputScaler.transform(self.X_train),
                self.outputScaler.transform(self.y_train),
            )

    def predict(self, X, y=None):
        if self.modelType == "RNN":
            test_generator = TimeseriesGenerator(
                self.inputScaler.transform(X),
                self.outputScaler.transform(y),
                length = self.args.enrolWindow,
                sampling_rate = 1,
                batch_size = self.args.batchSize
            )
            return self.outputScaler.inverse_transform(
                self.model.predict(test_generator)
            )
        else:
            return self.outputScaler.inverse_transform(
                self.model.predict(
                    self.inputScaler.transform(X)
                )
            )

    def save(self, directory, name):
        if self.args:
            self.model.save(directory + name + ".h5")

def ensembleModel(params, models):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    return EnsembleModel(models, X, Y, name=name)

def kerasLSTMSingleLayerLeaky(params, units=128, dropout=0.1, alpha=0.5):
    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = params['args']
    model = Sequential()
    model.add(LSTM(units, input_shape=(args.enrolWindow, X_train.shape[1])))
    model.add(LeakyReLU(alpha=alpha)) 
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1]))
    return MachinLearningModel(model, X_train, y_train, args=args, modelType="RNN", name=name)

def kerasLSTMMultiLayer(params, units=[50, 100], dropoutRate=0.2):
    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = params['args']
    model = Sequential()
    model.add(LSTM(units[0], return_sequences=True, input_shape=(args.enrolWindow, X_train.shape[1])))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(units[1], return_sequences=False))
    model.add(Dropout(dropoutRate))
    model.add(Dense(y_train.shape[1]))
    return MachinLearningModel(model, X_train, y_train, args=args, modelType="RNN", name=name)

def kerasLSTMSingleLayer(params, units=128, dropout=0.3, recurrentDropout=0.3):
    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = params['args']
    input_layer = Input(shape=(None,X_train.shape[-1]))
    layer_1 = layers.LSTM(units,
                         dropout = dropout,
                         recurrent_dropout = recurrentDropout,
                         return_sequences = True)(input_layer, training=True)

    output_layer = layers.Dense(y_train.shape[-1])(layer_1)
    
    model = Model(input_layer, output_layer) 
    return MachinLearningModel(model, X_train, y_train, args=args, modelType="RNN", name=name)

def kerasSequentialRegressionModel(params, structure):
    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = params['args']

    model = Sequential()

    firstLayerNeurons, firstLayerActivation = structure[0]
    model.add(Dense(firstLayerNeurons, input_dim=X_train.shape[1], activation=firstLayerActivation))
    
    for neurons, activation in structure[1:]:
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(y_train.shape[1], activation='linear'))

    return MachinLearningModel(model, X_train, y_train, args=args, modelType="MLP", name=name)

def kerasSequentialRegressionModelWithRegularization(params, structure, l1_rate=0.01, l2_rate=0.01, ):
    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = params['args']

    model = Sequential()

    firstLayerNeurons, firstLayerActivation = structure[0]
    model.add(Dense(firstLayerNeurons, input_dim=X_train.shape[1], activation=firstLayerActivation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate)))
    
    for neurons, activation in structure[1:]:
        model.add(Dense(neurons, activation=activation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate)))
    
    model.add(Dense(y_train.shape[1], activation='linear'))

    return MachinLearningModel(model, X_train, y_train, args=args, modelType="MLP", name=name)

def sklearnSVM(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = LinearSVR()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnDecisionTree(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = DecisionTreeRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnAdaBoost(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = AdaBoostRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnBagging(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = BaggingRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnGradientBoosting(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = GradientBoostingRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnRandomForest(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = RandomForestRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnMLP(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = MLPRegressor(early_stopping=True)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

# apparently not how RBMs work...
# look into it :) 
def sklearnRBM(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = BernoulliRBM()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnLinear(params):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = LinearRegression()
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnLasso(params, alpha=0.1):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = Lasso(alpha=alpha)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnLassoCV(params, alphas=(0.1, 1.0, 10.0)):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = LassoCV(alphas=alphas)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnRidge(params, alpha=1.0):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = Ridge(alpha=alpha)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnRidgeCV(params, alphas=(0.1, 1.0, 10.0)):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = RidgeCV(alphas=alphas)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnElasticNet(params, alpha=1.0, l1_ratio=0.5):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)

def sklearnElasticNetCV(params, alphas=None, l1_ratio=0.5):
    X = params['X_train']
    Y = params['y_train']
    name = params['name']
    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
    return MachinLearningModel(model, X, Y, modelType="Linear", name=name)
