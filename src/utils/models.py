from sklearn.linear_model import (ElasticNet, ElasticNetCV, LinearRegression, Lasso, LassoCV, Ridge, RidgeCV)
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1

from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU

class MachinLearningModel():
    def __init__(self, model, X_train, y_train, args=None):
        self.model = model
        self.args = args
        self.X_train = X_train
        self.y_train = y_train

    def train(self):
        if self.args:
            loss, optimizer, metrics, epochs, batchSize, verbose, callbacks = self.args

            model = self.model
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            model.fit(self.X_train,
                    self.y_train,
                    epochs=epochs,
                    batch_size=batchSize,
                    verbose=verbose,
                    callbacks=callbacks,
                    )
            return model
        else:
            model = self.model
            model.fit(self.X_train, self.y_train)
            return model

def kerasLSTMSingleLayerLeaky(train_X, y_train, units=128, dropout=0.1, alpha=0.5):
    model = Sequential()
    model.add(LSTM(units, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LeakyReLU(alpha=alpha)) 
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1]))
    return model

def kerasLSTMMultiLayer(train_X, y_train, units=[50, 100], dropoutRate=0.2):
    model = Sequential()
    model.add(LSTM(units[0], return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(units[1], return_sequences=False))
    model.add(Dropout(dropoutRate))
    model.add(Dense(y_train.shape[1]))
    return model

def kerasLSTMSingleLayer(train_X, y_train, units=128, dropout=0.3, recurrentDropout=0.3): 
    input_layer = Input(shape=(None,train_X.shape[-1]))
    layer_1 = layers.LSTM(units,
                         dropout = dropout,
                         recurrent_dropout = recurrentDropout,
                         return_sequences = True)(input_layer, training=True)

    output_layer = layers.Dense(y_train.shape[-1])(layer_1)
    
    model = Model(input_layer, output_layer) 
    return model

def kerasSequentialRegressionModel(X_train, y_train, layers, args):
    loss, optimizer, metrics, epochs, batchSize, verbose, callbacks = args
    
    model = Sequential()

    firstLayerNeurons, firstLayerActivation = layers[0]
    model.add(Dense(firstLayerNeurons, input_dim=X_train.shape[1], activation=firstLayerActivation))
    
    for neurons, activation in layers[1:]:
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(y_train.shape[1], activation='linear'))

    return MachinLearningModel(model, X_train, y_train, args)

def kerasSequentialRegressionModelWithRegularization(X_train, y_train, layers, args, l1_rate=0.01, l2_rate=0.01, ):
    loss, optimizer, metrics, epochs, batchSize, verbose, callbacks = args
    
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

    return MachinLearningModel(model, X_train, y_train, args)

def sklearnSVM(X, Y):
    model = LinearSVR()
    return MachinLearningModel(model, X, Y)

def sklearnDecisionTree(X, Y):
    model = DecisionTreeRegressor()
    return MachinLearningModel(model, X, Y)

def sklearnAdaBoost(X, Y):
    model = AdaBoostRegressor()
    return MachinLearningModel(model, X, Y)

def sklearnBagging(X, Y):
    model = BaggingRegressor()
    return MachinLearningModel(model, X, Y)

def sklearnGradientBoosting(X, Y):
    model = GradientBoostingRegressor()
    return MachinLearningModel(model, X, Y)

def sklearnRandomForest(X, Y):
    model = RandomForestRegressor()
    return MachinLearningModel(model, X, Y)

def sklearnMLP(X, Y):
    model = MLPRegressor(early_stopping=True)
    return MachinLearningModel(model, X, Y)

# apparently not how RBMs work...
# look into it :) 
def sklearnRBM(X, Y):
    model = BernoulliRBM()
    return MachinLearningModel(model, X, Y)

def sklearnLinear(X, Y):
    model = LinearRegression()
    return MachinLearningModel(model, X, Y)

def sklearnLasso(X, Y, alpha=0.1):
    model = Lasso(alpha=alpha)
    return MachinLearningModel(model, X, Y)

def sklearnLassoCV(X, Y, alphas=(0.1, 1.0, 10.0)):
    model = LassoCV(alphas=alphas)
    return MachinLearningModel(model, X, Y)

def sklearnRidge(X, Y, alpha=1.0):
    model = Ridge(alpha=alpha)
    return MachinLearningModel(model, X, Y)

def sklearnRidgeCV(X, Y, alphas=(0.1, 1.0, 10.0)):
    model = RidgeCV(alphas=alphas)
    return MachinLearningModel(model, X, Y)

def sklearnElasticNet(X, Y, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    return MachinLearningModel(model, X, Y)

def sklearnElasticNetCV(X, Y, alphas=None, l1_ratio=0.5):
    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
    return MachinLearningModel(model, X, Y)
