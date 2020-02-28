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
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs import (getConfig)

import matplotlib.pyplot as plt

EPOCHS = 4000
BATCH_SIZE = 128
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'relu'
LOSS = 'binary_crossentropy'
OPTIMIZER = 'adadelta'

def getModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(50, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(20, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

def main(filename):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = readDataFile(filename)
    df = getDataWithTimeIndex(df)
    df = df.dropna()

    traintime, testtime, validtime = timestamps

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    start_train, end_train = traintime
    start_test, end_test = testtime

    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    df_test = utilities.getDataByTimeframe(df, start_test, end_test)

    train = df_train.values
    test = df_test.values

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    scaler = MinMaxScaler(feature_range=(0,1))
    #scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    encoding_dim = 3

    input_d = Input(shape=(train.shape[1],))
    encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_d)
    decoded = Dense(train.shape[1], activation='linear')(encoded)

    autoencoder = Model(input_d, decoded)

    encoder = Model(input_d, encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(loss=LOSS, optimizer=OPTIMIZER)
    autoencoder.fit(train,
              train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=VERBOSE,
              callbacks=[callback],
              )
    
    encoded_data = encoder.predict(test)
    decoded_data = decoder.predict(encoded_data)

    encoded_data_train = encoder.predict(train)
    decoded_data_train = decoder.predict(encoded_data_train)

    print(test)
    test_sum = np.sum(np.absolute(test), axis=1)
    decode_sum = np.sum(np.absolute(decoded_data), axis=1)
    print(test_sum - decode_sum)
    
    #plt.show()


# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)
