from sklearn.metrics import (
    r2_score,
    mean_squared_log_error,
    mean_squared_error,
    mean_absolute_error,
    max_error
)

import numpy as np

np.random.seed(100)

def calculateR2Score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

def calculateMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

def calculateMAE(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def calculateMaxError(y_true, y_pred):
    if len(y_true.shape) > 1 and y_true.shape and y_true.shape[1] > 1:
        maxerror = None
    else:
        maxerror = max_error(y_true, y_pred)
    return maxerror

def calculateMetrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    if len(y_true.shape) > 1 and y_true.shape and y_true.shape[1] > 1:
        maxerror = None
    else:
        maxerror = max_error(y_true, y_pred)
    return [r2, mse, mae, maxerror]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

def relu_vectorized(x):
    return np.vectorize(relu)

def leaky_relu(x, a):
    return np.maximum(x, a*x)

def leaky_relu_vectorized(x, a):
    return np.vectorize(leaky_relu)

def elu(x, a):
    if x >= 0:
        return x
    else:
        return a*(np.exp(x) - 1)
