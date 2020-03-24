import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

from sklearn.linear_model import (ElasticNet, ElasticNetCV, LinearRegression, Lasso, LassoCV, Ridge, RidgeCV)
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.engine.input_layer import Input
from keras.regularizers import l2, l1
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks.callbacks import ModelCheckpoint
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU
from copy import deepcopy

import pickle
import numpy as np
import tensorflow as tf
from modelFuncs import getRNNSplit

np.random.seed(100)
tf.random.set_seed(100)

CURRENT_MODEL_WEIGHTS_FILEPATH = ROOT_PATH + '/src/ml/trained_models/'

class Args():
    def __init__(self, args):
        self.activation = args['activation']
        self.loss = args['loss']
        self.optimizer = args['optimizer']
        self.metrics = args['metrics']
        self.epochs = args['epochs']
        self.batchSize = args['batchSize']
        self.verbose = args['verbose']
        self.callbacks= args['callbacks']
        self.enrolWindow = args['enrolWindow']
        self.validationSize = args['validationSize']
        self.testSize = args['testSize']

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
        checkpoint_path = CURRENT_MODEL_WEIGHTS_FILEPATH + "_".join(self.name.split(" "))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        weights_path = checkpoint_path + "/current_weights.h5"
        checkpoint = ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        ),

        if self.modelType == "RNN":
            """
            THIS CODE CAN BE USED IF GENERATORS ARE DESIRED
            NB: not suitable for heat exchanger data,
                because the validation data will not have
                the same properties as the training data

            X_t, X_v, y_t, y_v = train_test_split(self.X_train, self.y_train, test_size=0.2, shuffle=False)
            validation_generator = TimeseriesGenerator(
                self.inputScaler.transform(X_v),
                self.outputScaler.transform(y_v),
                length = self.args.enrolWindow,
                sampling_rate = 1,
                batch_size = self.args.batchSize
            )
            train_generator = TimeseriesGenerator(
                self.inputScaler.transform(X_t),
                self.outputScaler.transform(y_t),
                length = self.args.enrolWindow,
                sampling_rate = 1,
                batch_size = self.args.batchSize
            )
            self.model.compile(
                loss = self.args.loss,
                optimizer = self.args.optimizer,
                metrics = self.args.metrics
            )
            history = self.model.fit_generator(
                train_generator,
                epochs = self.args.epochs,
                verbose = self.args.verbose,
                callbacks = [*self.args.callbacks, *checkpoint],
                validation_data = validation_generator,
            )
            """
            # Own implementation of train-val split
            #     for RNN data. Uses every n'th sample
            #     of data for validation
            X_t, X_v, y_t, y_v = getRNNSplit(
                self.inputScaler.transform(self.X_train),
                self.outputScaler.transform(self.y_train),
                self.args.enrolWindow,
            )
            self.model.compile(
                loss = self.args.loss,
                optimizer = self.args.optimizer,
                metrics = self.args.metrics
            )
            history = self.model.fit(
                X_t,
                y_t,
                epochs = self.args.epochs,
                batch_size = self.args.batchSize,
                verbose = self.args.verbose,
                callbacks = [*self.args.callbacks, *checkpoint],
                validation_data = (X_v, y_v),
            )
            self.history = history.history
            self.model.load_weights(weights_path)
        elif self.modelType == "MLP":
            self.model.compile(
                loss = self.args.loss,
                optimizer = self.args.optimizer,
                metrics = self.args.metrics
            )
            history = self.model.fit(
                self.inputScaler.transform(self.X_train),
                self.outputScaler.transform(self.y_train),
                epochs = self.args.epochs,
                batch_size = self.args.batchSize,
                verbose = self.args.verbose,
                callbacks = [*self.args.callbacks, *checkpoint],
                validation_split = self.args.validationSize,
            )
            self.history = history.history
            self.model.load_weights(weights_path)
        else:
            history = self.model.fit(
                self.inputScaler.transform(self.X_train),
                self.outputScaler.transform(self.y_train),
            )
            self.history = None

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

    def predictMultiple(self, X, y, numberOfPredictions=20):
        if self.modelType == "RNN":
            predictions = np.zeros((numberOfPredictions, (y.shape[0] - self.args.enrolWindow), y.shape[1]))
            for i in range(numberOfPredictions):
                predictions[i] = self.predict(X, y)

            mean = np.array([np.mean(predictions[:,:,i], axis=0) for i in range(y.shape[1])]).T
            standarddev = np.array([np.std(predictions[:,:,i], axis=0) for i in range(y.shape[1])]).T
        
            return [predictions, mean, standarddev]
        else:
            return None

    def save(self, directory, name):
        if self.args:
            self.model.save(directory + name + ".h5")
            with open(directory + name + ".pickle", 'wb') as file_pi:
                pickle.dump(self.history, file_pi)


class AutoencoderModel():
    def __init__(self, model, X_train, args=None, modelType="AUTOENCODER", scaler="standard", name=None):
        if scaler == "standard":
            inputScaler = StandardScaler()
        else:
            inputScaler = MinMaxScaler()
        
        inputScaler.fit(X_train)

        self.model = model
        self.X_train = X_train
        self.args = args
        self.name = name
        self.history = None
        self.inputScaler = inputScaler
        self.modelType = modelType

    def train(self):
        self.model.compile(
            loss = self.args.loss,
            optimizer = self.args.optimizer,
            metrics = self.args.metrics
        )
        history = self.model.fit(
            self.inputScaler.transform(self.X_train),
            self.inputScaler.transform(self.X_train),
            epochs = self.args.epochs,
            batch_size = self.args.batchSize,
            verbose = self.args.verbose,
            callbacks = self.args.callbacks,
            validation_split = self.args.validationSize,
        )
        self.history = history.history

    def predict(self, X, y=None):
        return self.inputScaler.inverse_transform(
            self.model.predict(
                self.inputScaler.transform(X)
            )
        )

    def save(self, directory, name):
        if self.args:
            self.model.save(directory + name + ".h5")
            with open(directory + name + ".pickle", 'wb') as file_pi:
                pickle.dump(self.history, file_pi)

def ensembleModel(
    params,
    models,
    ):

    X = params['X_train']
    Y = params['y_train']
    name = params['name']

    return EnsembleModel(
        models,
        X,
        Y,
        name=name,
    )

def kerasLSTM(
    params,
    layers=[128],
    dropout=0.0,
    recurrentDropout=0.0,
    alpha=None,
    training=False,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])
    input_layer = Input(shape=(None,X_train.shape[-1]))

    if len(layers) > 1:
        firstLayerUnits = layers[0]
        layer_1 = LSTM(firstLayerUnits,
                            activation = args.activation,
                            dropout = dropout,
                            recurrent_dropout = recurrentDropout,
                            return_sequences = True)(input_layer, training=training)
        if alpha is not None:
            layer_1 = LeakyReLU(alpha=alpha)(layer_1)
        for i, layerUnits in enumerate(layers[1:]):
            layer_1 = LSTM(layerUnits,
                            activation = args.activation,
                            dropout = dropout,
                            recurrent_dropout = recurrentDropout,
                            return_sequences = True if (i < len(layers) - 2) else False)(layer_1, training=training)
            if alpha is not None:
                layer_1 = LeakyReLU(alpha=alpha)(layer_1)
    else:
        firstLayerUnits = layers[0]
        layer_1 = LSTM(firstLayerUnits,
                            activation = args.activation,
                            dropout = dropout,
                            return_sequences = False,
                            recurrent_dropout = recurrentDropout)(input_layer, training=training)
        if alpha is not None:
            layer_1 = LeakyReLU(alpha=alpha)(layer_1)

    output_layer = Dense(
        y_train.shape[-1],
        activation='linear')(layer_1)
    
    model = Model(input_layer, output_layer)

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="RNN",
        name=name,
    )

def kerasGRU(
    params,
    layers=[128],
    dropout=0.0,
    recurrentDropout=0.0,
    alpha=None,
    training=False,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])
    input_layer = Input(shape=(None,X_train.shape[-1]))

    if len(layers) > 1:
        firstLayerUnits = layers[0]
        layer_1 = GRU(
            firstLayerUnits,
            activation = args.activation,
            dropout = dropout,
            recurrent_dropout = recurrentDropout,
            return_sequences = True)(input_layer, training=training)
        if alpha is not None:
            layer_1 = LeakyReLU(alpha=alpha)(layer_1)
        for layerUnits in layers[1:]:
            layer_1 = GRU(
                layerUnits,
                activation = args.activation,
                dropout = dropout,
                recurrent_dropout = recurrentDropout,
                return_sequences = False)(layer_1, training=training)
            if alpha is not None:
                layer_1 = LeakyReLU(alpha=alpha)(layer_1)
    else:
        firstLayerUnits = layers[0]
        layer_1 = GRU(
            firstLayerUnits,
            activation = args.activation,
            dropout = dropout,
            recurrent_dropout = recurrentDropout)(input_layer, training=training)
        if alpha is not None:
            layer_1 = LeakyReLU(alpha=alpha)(layer_1)

    output_layer = Dense(
        y_train.shape[-1],
        activation='linear')(layer_1)
    
    model = Model(input_layer, output_layer)

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="RNN",
        name=name,
    )

def kerasMLP(
    params,
    structure,
    dropout=None,
    l1_rate=0.0,
    l2_rate=0.0,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])

    model = Sequential()

    firstLayerNeurons, firstLayerActivation = structure[0]
    model.add(
        Dense(
            firstLayerNeurons,
            input_dim=X_train.shape[1],
            activation=firstLayerActivation,
            kernel_regularizer=l2(l2_rate),
            activity_regularizer=l1(l1_rate),
        )
    )
    if dropout is not None:
        model.add(Dropout(dropout))

    for neurons, activation in structure[1:]:
        model.add(
            Dense(
                neurons,
                activation=activation,
            )
        )
        if dropout is not None:
            model.add(Dropout(dropout))
    
    model.add(
        Dense(
            y_train.shape[1],
            activation='linear',
        )
    )

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="MLP",
        name=name,
    )

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

def autoencoder_Dropout(params, dropout=0.2, encodingDim=3):
    X = params['X_train']
    name = params['name']
    args = Args(params['args'])

    if encodingDim > 3:
        encodingDim = 3

    input_d = Input(shape=(X.shape[1],))
    encoded = Dense(6, activation='tanh')(input_d)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(5, activation='tanh')(encoded)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(4, activation='tanh')(encoded)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(encodingDim, activation='tanh')(encoded)
    #encoded = Dropout(dropout)(encoded)
    decoded = Dense(4, activation='tanh')(encoded)
    #decoded = Dropout(dropout)(decoded)
    decoded = Dense(5, activation='tanh')(decoded)
    #decoded = Dropout(dropout)(decoded)
    decoded = Dense(6, activation='tanh')(decoded)
    #decoded = Dropout(dropout)(decoded)
    decoded = Dense(X.shape[1], activation='linear')(decoded)
    model = Model(input_d, decoded)
    return AutoencoderModel(model, X, args, modelType="AUTOENCODER", name=name)

def autoencoder_Regularized(params, l1_rate=10e-4, encodingDim=3):
    X = params['X_train']
    name = params['name']
    args = Args(params['args'])

    if encodingDim > 3:
        encodingDim = 3

    model = Sequential()
    model.add(Dense(X.shape[1]))
    model.add(Dense(6, activation='tanh', activity_regularizer=l1(l1_rate)))
    model.add(Dense(5, activation='tanh', activity_regularizer=l1(l1_rate)))
    model.add(Dense(4, activation='tanh', activity_regularizer=l1(l1_rate)))
    model.add(Dense(encodingDim, activation='tanh', activity_regularizer=l1(l1_rate)))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(5, activation='tanh'))
    model.add(Dense(6, activation='tanh'))
    model.add(Dense(X.shape[1], activation='linear'))
    return AutoencoderModel(model, X, args, modelType="AUTOENCODER", name=name)