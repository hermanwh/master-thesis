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
        self.MLmodel = sklearnLinear(train, self.y_train[self.maxEnrol:])
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
        self.MLmodel = sklearnLinear(train, self.y_train[self.maxEnrol:])
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
        for i, model in enumerate(self.models):
            if model.args:
                dirr = directory + name + '/'
                if not os.path.exists(dirr):
                    os.makedirs(dirr)
                model.save(dirr, str(i))

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

def ensembleModel(models, X_train, y_train):
    return EnsembleModel(models, X_train, y_train, )

def kerasLSTMSingleLayerLeaky(X_train, y_train, args, units=128, dropout=0.1, alpha=0.5):
    model = Sequential()
    model.add(LSTM(units, input_shape=(args.enrolWindow, X_train.shape[1])))
    model.add(LeakyReLU(alpha=alpha)) 
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1]))
    return MachinLearningModel(model, X_train, y_train, args=args, modelType="RNN")

def kerasLSTMMultiLayer(X_train, y_train, args, units=[50, 100], dropoutRate=0.2):
    model = Sequential()
    model.add(LSTM(units[0], return_sequences=True, input_shape=(args.enrolWindow, X_train.shape[1])))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(units[1], return_sequences=False))
    model.add(Dropout(dropoutRate))
    model.add(Dense(y_train.shape[1]))
    return MachinLearningModel(model, X_train, y_train, args=args, modelType="RNN")

def kerasLSTMSingleLayer(X_train, y_train, args, units=128, dropout=0.3, recurrentDropout=0.3):
    input_layer = Input(shape=(None,X_train.shape[-1]))
    layer_1 = layers.LSTM(units,
                         dropout = dropout,
                         recurrent_dropout = recurrentDropout,
                         return_sequences = True)(input_layer, training=True)

    output_layer = layers.Dense(y_train.shape[-1])(layer_1)
    
    model = Model(input_layer, output_layer) 
    return MachinLearningModel(model, X_train, y_train, args=args, modelType="RNN")

def kerasSequentialRegressionModel(X_train, y_train, args, layers):
    model = Sequential()

    firstLayerNeurons, firstLayerActivation = layers[0]
    model.add(Dense(firstLayerNeurons, input_dim=X_train.shape[1], activation=firstLayerActivation))
    
    for neurons, activation in layers[1:]:
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(y_train.shape[1], activation='linear'))

    return MachinLearningModel(model, X_train, y_train, args=args, modelType="MLP")

def kerasSequentialRegressionModelWithRegularization(X_train, y_train, args, layers, l1_rate=0.01, l2_rate=0.01, ):
    model = Sequential()

    firstLayerNeurons, firstLayerActivation = layers[0]
    model.add(Dense(firstLayerNeurons, input_dim=X_train.shape[1], activation=firstLayerActivation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate)))
    
    for neurons, activation in layers[1:]:
        model.add(Dense(neurons, activation=activation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate)))
    
    model.add(Dense(y_train.shape[1], activation='linear'))

    return MachinLearningModel(model, X_train, y_train, args=args, modelType="MLP")

def sklearnSVM(X, Y):
    model = LinearSVR()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnDecisionTree(X, Y):
    model = DecisionTreeRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnAdaBoost(X, Y):
    model = AdaBoostRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnBagging(X, Y):
    model = BaggingRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnGradientBoosting(X, Y):
    model = GradientBoostingRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnRandomForest(X, Y):
    model = RandomForestRegressor()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnMLP(X, Y):
    model = MLPRegressor(early_stopping=True)
    return MachinLearningModel(model, X, Y, modelType="Linear")

# apparently not how RBMs work...
# look into it :) 
def sklearnRBM(X, Y):
    model = BernoulliRBM()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnLinear(X, Y):
    model = LinearRegression()
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnLasso(X, Y, alpha=0.1):
    model = Lasso(alpha=alpha)
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnLassoCV(X, Y, alphas=(0.1, 1.0, 10.0)):
    model = LassoCV(alphas=alphas)
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnRidge(X, Y, alpha=1.0):
    model = Ridge(alpha=alpha)
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnRidgeCV(X, Y, alphas=(0.1, 1.0, 10.0)):
    model = RidgeCV(alphas=alphas)
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnElasticNet(X, Y, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    return MachinLearningModel(model, X, Y, modelType="Linear")

def sklearnElasticNetCV(X, Y, alphas=None, l1_ratio=0.5):
    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
    return MachinLearningModel(model, X, Y, modelType="Linear")

def printModelSummary(model):
    if hasattr(model, "summary"):
        # Keras Model object
        print(model.summary())
    elif hasattr(model, "model"):
        # MachineLearningModel object
        if hasattr(model.model, "summary"):
            # MachineLearningModel.model will be a Keras Model object
            printModelSummary(model.model)
    elif hasattr(model, "models"):
        # EnsembleModel object
        print("Model is of type Ensemble Model")
        print("Sub model summaries will follow")
        print("-------------------------------")
        for mod in model.models:
            # EnsembleModel.models will be a list of MachineLearningModels
            printModelSummary(mod)
    else:
        print("Simple models have no summary")
    
def printModelWeights(model):
    if hasattr(model, "summary"):
        # Keras Model object
        for layer in model.layers: print(layer.get_config(), layer.get_weights())
    elif hasattr(model, "model"):
        # MachineLearningModel object
        if hasattr(model.model, "summary"):
            # MachineLearningModel.model will be a Keras Model object
            printModelSummary(model.model)
    elif hasattr(model, "models"):
        # EnsembleModel object
        print("Model is of type Ensemble Model")
        print("Sub model summaries will follow")
        print("-------------------------------")
        for mod in model.models:
            # EnsembleModel.models will be a list of MachineLearningModels
            printModelSummary(mod)
    else:
        if hasattr(model, "get_params"):
            print(model.get_params())
        else:
            print("No weights found")