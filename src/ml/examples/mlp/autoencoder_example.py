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
import tensorflow as tf
import numpy as np
from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)
from models import (kerasSequentialRegressionModel,
                    sklearnMLP,
                    sklearnLinear,
                    sklearnRidgeCV
                    )

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs import (getConfig)

import matplotlib.pyplot as plt

EPOCHS = 4000
BATCH_SIZE = 32
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'tanh'
OPTIMIZER = 'adam'

def getModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(50, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(20, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

def main(filename):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)
    
    labelNames = {
        '20TT001': 'Gas side inlet temperature',
        '20PT001': 'Gas side inlet pressure',
        '20FT001': 'Gas side flow',
        '20TT002': 'Gas side outlet temperature',
        '20PDT001': 'Gas side pressure differential',
        '50TT001': 'Cooling side inlet temperature',
        '50PT001': 'Cooling side inlet pressure',
        '50FT001': 'Cooling side flow',
        '50TT002': 'Cooling side outlet temperature',
        '50PDT001': 'Cooling side pressure differential',
        '50TV001': 'CM Valve opening',
    }

    irrelevantColumns = [
        '20TT001',
        '20PT001',
        '50TT001',
        '50PT001',
        '20TT002',
    ]

    columns = list(labelNames.keys())
    relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))
    """
    
    labelNames = {
        'Date':'Date',
        'FYN0111': 'Gasseksport rate',
        'FT0111': 'Gasseksport molvekt',
        'TT0102_MA_Y': 'Varm side A temperatur inn',
        'TIC0101_CA_YX': 'Varm side A temperatur ut',
        'TT0104_MA_Y': 'Varm side B temperatur inn',
        'TIC0103_CA_YX': 'Varm side B temperatur ut',
        'TT0106_MA_Y': 'Varm side C temperatur inn',
        'TIC0105_CA_YX': 'Varm side C temperatur ut',
        'TI0115_MA_Y': 'Scrubber temperatur ut',
        'PDT0108_MA_Y': 'Varm side A trykkfall',
        'PDT0119_MA_Y': 'Varm side B trykkfall',
        'PDT0118_MA_Y': 'Varm side C trykkfall',
        'PIC0104_CA_YX': 'Innløpsseparator trykk',
        'TIC0425_CA_YX': 'Kald side temperatur inn',
        'TT0651_MA_Y': 'Kald side A temperatur ut',
        'TT0652_MA_Y': 'Kald side B temperatur ut',
        'TT0653_MA_Y': 'Kald side C temperatur ut',
        'TIC0101_CA_Y': 'Kald side A ventilåpning',
        'TIC0103_CA_Y': 'Kald side B ventilåpning',
        'TIC0105_CA_Y': 'Kald side C ventilåpning',
    }

    irrelevantColumns = [
        'FT0111',
        'TT0104_MA_Y',
        'TIC0103_CA_YX',
        'PDT0108_MA_Y',
        'PDT0119_MA_Y',
        'PDT0118_MA_Y',
        'TT0652_MA_Y',
    ]

    traintime = [["2017-08-05 00:00:00", "2018-06-01 00:00:00"]]
    testtime = ["2017-08-05 00:00:00", "2019-11-01 00:00:00"]
    validtime = ["2018-05-01 00:00:00", "2018-07-01 00:00:00"]

    columns = list(labelNames.keys())
    relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))
    """
    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    traintime, testtime, validtime = timestamps

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    start_train, end_train = traintime[0]
    start_test, end_test = testtime

    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    df_test = utilities.getDataByTimeframe(df, start_test, end_test)

    train = df_train.values
    test = df_test.values

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150)

    #scaler = MinMaxScaler(feature_range=(-2,1))
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    autoencoder1(train, test, callback, df_test)

def autoencoder1(train, test, callback, df_test):
    encoding_dim = 3
    dropoutRate = 0.2

    input_d = Input(shape=(train.shape[1],))
    encoded = Dense(6, activation=ACTIVATION)(input_d)
    encoded = Dropout(dropoutRate)(encoded)
    encoded = Dense(5, activation=ACTIVATION)(encoded)
    encoded = Dropout(dropoutRate)(encoded)
    encoded = Dense(4, activation=ACTIVATION)(encoded)
    encoded = Dropout(dropoutRate)(encoded)
    encoded = Dense(encoding_dim, activation=ACTIVATION)(encoded)
    encoded = Dropout(dropoutRate)(encoded)
    decoded = Dense(4, activation=ACTIVATION)(encoded)
    decoded = Dropout(dropoutRate)(decoded)
    decoded = Dense(5, activation=ACTIVATION)(decoded)
    decoded = Dropout(dropoutRate)(decoded)
    decoded = Dense(6, activation=ACTIVATION)(decoded)
    decoded = Dropout(dropoutRate)(decoded)
    decoded = Dense(train.shape[1], activation='linear')(decoded)

    autoencoder = Model(input_d, decoded)

    autoencoder.compile(loss='mse', optimizer='adam')
    autoencoder.fit(
        train,
        train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        callbacks=[callback],
        validation_split=0.2,
    )

    encoder = Model(input_d, encoded)
    
    encoded_data = encoder.predict(test)

    encoded_data_train = encoder.predict(train)
    print("encoded data", encoded_data_train)

    pred_test = autoencoder.predict(test)

    #print(test)
    test_sum = np.sum(np.absolute(test), axis=1)
    decode_sum = np.sum(np.absolute(pred_test), axis=1)
    #print(test_sum - decode_sum)
    
    indexx = df_test.index

    for i in range(test.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        ax.plot(indexx, pred_test[:, i], color='red')
        ax.plot(indexx, test[:, i], color='blue')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_ylabel('Value', fontsize=12)

        ax.set_title(df_test.columns[i], fontsize=16)

    plt.show()

    for i in range(test.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        ax.plot(indexx, test[:, i] - pred_test[:, i], color='red')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_ylabel('Deviation', fontsize=12)

        ax.set_title(df_test.columns[i], fontsize=16)

    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    ax.plot(indexx, np.average((test - pred_test)**2,axis=1), color='red')
    ax.set_xlabel('Date', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_ylabel('Error', fontsize=12)

    ax.set_title('Reconstruction error', fontsize=16)

    plt.show()


def autoencoder2(train, test, callback, df_test):
    dropoutRate = 0.2

    model = Sequential()

    model.add(Dense(train.shape[1]))
    model.add(Dropout(dropoutRate))
    model.add(Dense(6, activation=ACTIVATION, activity_regularizer=regularizers.l1(10e-4)))
    model.add(Dropout(dropoutRate))
    model.add(Dense(5, activation=ACTIVATION, activity_regularizer=regularizers.l1(10e-4)))
    model.add(Dropout(dropoutRate))
    model.add(Dense(4, activation=ACTIVATION, activity_regularizer=regularizers.l1(10e-4)))
    model.add(Dropout(dropoutRate))
    model.add(Dense(3, activation=ACTIVATION, activity_regularizer=regularizers.l1(10e-4)))
    model.add(Dropout(dropoutRate))
    model.add(Dense(4, activation=ACTIVATION))
    model.add(Dropout(dropoutRate))
    model.add(Dense(5, activation=ACTIVATION))
    model.add(Dropout(dropoutRate))
    model.add(Dense(6, activation=ACTIVATION))
    model.add(Dropout(dropoutRate))
    model.add(Dense(train.shape[1], activation='linear'))
    
    model.compile(loss='mse', optimizer=OPTIMIZER)
    model.fit(train,
              train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=VERBOSE,
              callbacks=[callback],
              validation_split=0.2,
              )

    pred_train = model.predict(train)
    pred_test = model.predict(test)

    print(train - pred_train)
    print(test - pred_test)

    print(np.sum((train - pred_train)**2, axis=1))

    indexx = df_test.index

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    ax.plot(indexx, np.average(((test - pred_test)/test)**2,axis=1), color='red')
    ax.set_xlabel('Date', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_ylabel('Error', fontsize=12)

    ax.set_title('Reconstruction error', fontsize=16)

    plt.show()

# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)

