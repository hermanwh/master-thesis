from sklearn.linear_model import (ElasticNet, ElasticNetCV, LinearRegression, Lasso, LassoCV, Ridge, RidgeCV)
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1

def kerasSequentialRegressionModel(layers, inputDim, outputDim=1):
    model = Sequential()

    firstLayerNeurons, firstLayerActivation = layers[0]
    model.add(Dense(firstLayerNeurons, input_dim=inputDim, activation=firstLayerActivation))
    
    for neurons, activation in layers[1:]:
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(outputDim, activation='linear'))
    return model

def kerasSequentialRegressionModelWithRegularization(layers, inputDim, outputDim=1, l1_rate=0.01, l2_rate=0.01):
    model = Sequential()

    firstLayerNeurons, firstLayerActivation = layers[0]
    model.add(Dense(firstLayerNeurons, input_dim=inputDim, activation=firstLayerActivation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate)))
    
    for neurons, activation in layers[1:]:
        model.add(Dense(neurons, activation=activation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate)))
    
    model.add(Dense(outputDim, activation='linear'))
    return model


def basicSequentialModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(50, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(20, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

def sklearnSVM(X, Y):
    model = LinearSVR()
    model.fit(X, Y)
    return model

def sklearnDecisionTree(X, Y):
    model = DecisionTreeRegressor()
    model.fit(X, Y)
    return model

def sklearnAdaBoost(X, Y):
    model = AdaBoostRegressor()
    model.fit(X, Y)
    return model

def sklearnBagging(X, Y):
    model = BaggingRegressor()
    model.fit(X, Y)
    return model

def sklearnGradientBoosting(X, Y):
    model = GradientBoostingRegressor()
    model.fit(X, Y)
    return model

def sklearnRandomForest(X, Y):
    model = RandomForestRegressor()
    model.fit(X, Y)
    return model

def sklearnMLP(X, Y):
    model = MLPRegressor(early_stopping=True)
    model.fit(X, Y)
    return model

# apparently not how RBMs work...
# look into it :) 
def sklearnRBM(X, Y):
    model = BernoulliRBM()
    model.fit(X, Y)
    return model

def sklearnLinear(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

def sklearnLasso(X, Y, alpha=0.1):
    model = Lasso(alpha=alpha)
    model.fit(X, Y)
    return model

def sklearnLassoCV(X, Y, alphas=(0.1, 1.0, 10.0)):
    model = LassoCV(alphas=alphas)
    model.fit(X, Y)
    return model

def sklearnRidge(X, Y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model

def sklearnRidgeCV(X, Y, alphas=(0.1, 1.0, 10.0)):
    model = RidgeCV(alphas=alphas)
    model.fit(X, Y)
    return model

def sklearnElasticNet(X, Y, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, Y)
    return model

def sklearnElasticNetCV(X, Y, alphas=None, l1_ratio=0.5):
    model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
    model.fit(X, Y)
    return model
