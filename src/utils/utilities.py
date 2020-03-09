import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error

INTERPOLDEG = 3

def getColorScheme():
    return {
        'b1':"#0051FF",
        'b2':"#007CFF",
        'b3':"#4DA3FE",
        'r1':"#FF0101",
        'r2':"#FF3B3B",
        'r3':"#EB7D00",
        'r4':"#EBCF00",
    }

def testForGPU():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

def dropIrrelevantColumns(df, args):
    relevantColumns, columnDescriptions = args

    print("Columns before removal")
    printColumns(df, columnDescriptions)
    printHorizontalLine()

    dfcolumns = df.columns
    for column in dfcolumns:
        if column not in relevantColumns:
            df = df.drop(column, axis=1)

    print("Columns after removal")
    printColumns(df, columnDescriptions)
    printHorizontalLine()
    
    return df

def printData(df):
    print(df)
    printHorizontalLine()

def printDataByTimeframe(df, start, end):
    df = getDataByTimeframe(df, start, end)
    printData(df)

def printColumns(df, columnDescriptions):
    printHorizontalLine()
    print("Dataset columns:")
    for i, column in enumerate(df.columns):
        if columnDescriptions is not None and column in columnDescriptions:
            print("Col.", i, ":", column, "-", columnDescriptions[column])
        else:
            print("Col.", i, ":", column)

def prettyPrint(data, precision, suppress):
    print(np.array_str(data, precision=precision, suppress_small=suppress))

def readDataFile(filename):
    ext = filename[-4:]
    if ext == '.csv':
        df = pd.read_csv(filename)
        if 'Date' in df.columns:
            print(df['Date'])
            df['Date'] = df['Date'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        elif 'time' in df.columns:
            df['Date'] = df['time'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df.drop('time', axis=1)
    elif ext == '.xls':
        df = pd.read_excel(filename)
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        elif 'time' in df.columns:
            df['Date'] = df['time'].apply(lambda x: x.split('+')[0])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df.drop('time', axis=1)
    else:
        print("Could not load data from file")
        print("Use .csv or .xls format")
        df = False
    return df

def getDataWithTimeIndex(df, dateColumn='Date'):
    if dateColumn in df.columns:
        print("Date index set")
        df = df.set_index(dateColumn, inplace=False)
    else:
        print("No date column")
    return df

def getDataByTimeframe(df, start, end):
    print(df)
    print("Finding data between", start, "and", end)
    printHorizontalLine()
    df = df.loc[start:end]
    return df

def saveKerasModel(model, loc, name):
    print("Saving model")
    model.save(loc + '/' + name + '.h5')
    print("Model saved")
    printHorizontalLine()

def printEmptyLine():
    print("")

def printHorizontalLine():
    print("-------------------------------------------")

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
