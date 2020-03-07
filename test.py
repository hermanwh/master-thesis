


import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import utilities
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
from utilities import (readDataFile,
                       getDataWithTimeIndex,
                       getDataByTimeframe,
                       printEmptyLine,
                       plotData,
                       plotDataColumnSingle
                       )
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs import (getConfig, getConfigDirs)
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

import matplotlib.pyplot as plt

EPOCHS = 4000
BATCH_SIZE = 128
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

def main(filename):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = readDataFile(filename)
    df = getDataWithTimeIndex(df)
    df = df.dropna()

    traintime, testtime, validtime = timestamps

    start_train, end_train = traintime
    start_test, end_test = testtime

    df['FYN0111'] = df['FYN0111'].apply(lambda x: x/2)

    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    df_test = utilities.getDataByTimeframe(df, start_test, end_test)

    targetColumns = [
        'TT0651_MA_Y'
    ]

    featureColumns = [
        'TT0102_MA_Y',
        'PIC0104_CA_YX',
        'FYN0111',
        'TIC0101_CA_YX',
        'TIC0425_CA_YX',
    ]

    X_train = df_train[featureColumns].values
    y_train = df_train[targetColumns].values

    X_test = df_test[featureColumns].values
    y_test = df_test[targetColumns].values

    #scaler = MinMaxScaler(feature_range=(0,1))
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = load_model('C:/Users/HHOR/Apps/master-thesis/src/ml/trained_models/D/model.h5')

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_metrics = utilities.calculateMetrics(y_train, pred_train)
    test_metrics = utilities.calculateMetrics(y_test, pred_test)

    print(train_metrics)
    print(test_metrics)

    for i in range(y_train.shape[1]):
        utilities.plotColumns(
            df_test,
            plt,
            [
                [
                    'Deviation', 
                    targetColumns[i],
                    y_test[:, i] - pred_test[:, i],
                    'darkgreen',
                    0.5,
                ]
            ],
            desc="Deviation, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
        )
        utilities.plotColumns(
            df_test,
            plt,
            [
                [
                    'Predictions',
                    targetColumns[i],
                    pred_test[:, i],
                    'darkgreen',
                    0.5,
                ],
                [
                    'Targets',
                    targetColumns[i],
                    y_test[:, i],
                    'red',
                    0.5,
                ]
            ],
            desc="Prediction vs. targets, ",
            columnDescriptions=labelNames,
            trainEndStr=end_train,
        )
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv targetCol
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)




"""

import requests

URL = "http://factpages.npd.no/Default.aspx?culture=nb-no&nav1=field&nav2=TableView%7cProduction%7cSaleable%7cMonthly"
URL = "https://www.ntnu.no/studier/emner/TDT4100"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
}


print(URL)

page = requests.get(URL, headers={'Accept-Encoding': None})

pprint(asd)

"""


"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

Y = np.array([0, 2, 1, 3, 5, 4, 7, 8, 6])
X = list(range(Y.shape[0]))

print(Y)
print(X)

print(len(Y))
print(len(X))

polynomialCoefficients = np.polyfit(X, Y, 3)
print(polynomialCoefficients)

polynomal = np.poly1d(polynomialCoefficients)
print(polynomal)

X_fine = np.linspace(X[0], X[-1], 1000)
print(X_fine)

func_vals = polynomal(X_fine)
print(func_vals)

fig,ax = plt.subplots()
ax.set_xlabel('X-verdi')
ax.set_ylabel('Y-verdi')
ax.set_title('Et plott')

ax.plot(X, Y, label='Plot of actual points', color="black")
ax.plot(X_fine, func_vals, label='Interpolated values', color="darkgreen")

func_vals = polynomal(X)
sinefunc = np.sin(X)
print(sinefunc)
print(func_vals)
joined = np.concatenate((sinefunc.reshape(-1, 1), func_vals.reshape(-1, 1)), axis=1)

model = LinearRegression().fit(joined, Y)


pred = model.predict(np.concatenate((np.sin(X_fine).reshape(-1, 1), polynomal(X_fine).reshape(-1, 1)), axis=1))

ax.plot(X_fine, pred, label='Prediction', color="red")

plt.show()

"""