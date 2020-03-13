#%%
import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import utilities
import plots
import metrics
import inspect
import tensorflow as tf
import numpy as np
from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)
from models import (kerasSequentialRegressionModel,
                    kerasSequentialRegressionModelWithRegularization,
                    sklearnMLP,
                    sklearnLinear,
                    sklearnRidgeCV
                    )
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs import (getConfig)
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

EPOCHS = 10000
BATCH_SIZE = 128
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

#%%
filename = "../master-thesis-db/datasets/B/data_20min.csv"
subdir = filename.split('/')[-2]
columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

traintime, testtime, validtime = timestamps

df = utilities.readDataFile(filename)
df = utilities.getDataWithTimeIndex(df)
df = df.dropna()

if relevantColumns is not None:
    df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

print(df.columns)