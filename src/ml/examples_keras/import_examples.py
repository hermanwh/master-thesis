
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras import optimizers
from keras.layers import LeakyReLU, ELU, ReLU
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.activations import linear, tanh, relu, elu
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn import decomposition
from sklearn import datasets

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor

from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.utils import shuffle

import time
import utilities
import sys
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D