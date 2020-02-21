import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.dates as mdates
import pyims
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error

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

def plotDataColumnSingle(df, plt, column, data, columnDescriptions=None, color='darkgreen'):
    fig, ax1 = plt.subplots()
    ax1.set_title("Plot")
    ax1.set_xlabel('Date')
    if columnDescriptions:
        ax1.set_ylabel(columnDescriptions[column])
    else:
        ax1.set_ylabel(column)
    ax1.plot(df.index, data, color=color, label="Data")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(1, axis='y')

    ax1.legend(loc='upper left')
    fig.autofmt_xdate()

def plotDataColumn(df, plt, column, pred, y, columnDescriptions=None, color1='darkgreen', color2='red'):
    fig, ax1 = plt.subplots()
    ax1.set_title("Targets vs predictions")
    color = color1
    ax1.set_xlabel('Date')
    if columnDescriptions:
        ax1.set_ylabel(columnDescriptions[column])
    else:
        ax1.set_ylabel(column)
    ax1.plot(df.index, pred, color=color, label="Predictions")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(1, axis='y')

    color = color2
    ax1.plot(df.index, y, color=color, label="Targets", alpha=0.5)

    ax1.legend(loc='upper left')
    fig.autofmt_xdate()

def plotData(df, plt, columnDescriptions=None, relevantColumns=None, color='darkgreen'):
    if relevantColumns is not None:
        columns = relevantColumns
    else:
        columns = df.columns

    for column in columns:
        if column != "Date":
            fig, ax1 = plt.subplots()
            ax1.set_title('Plot of dataset column ' + column)
            ax1.set_xlabel('Date')
            if columnDescriptions is not None and column in columnDescriptions:
                ax1.set_ylabel(column + " " + columnDescriptions[column], color=color)
            else:
                ax1.set_ylabel(column, color=color)
            ax1.plot(df.index, df[column], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(1, axis='y')

def plotDataByTimeframe(df, plt, start, end, columnDescriptions=None):
    df = getDataByTimeframe(df, start, end)
    for column in df.columns:
        if column != "Date":
            fig, ax1 = plt.subplots()
            ax1.set_title('Plot of dataset column ' + column)
            color = 'darkgreen'
            ax1.set_xlabel('Date')
            ax1.set_ylabel(columnDescriptions[column], color=color)
            ax1.plot(df.index, df[column], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(1, axis='y')
    plt.show()

def printData(df):
    print(df)
    printHorizontalLine()

def printDataByTimeframe(df, start, end):
    df = getDataByTimeframe(df, start, end)
    print(df)
    printHorizontalLine()

def saveKerasModel(model, loc, name):
    print("Saving model")
    model.save(loc + '/' + name + '.h5')
    print("Model saved")
    printHorizontalLine()

def compareR2Score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

def compareMetrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    maxerror = max_error(y_true, y_pred)
    return [r2, mse, mae, maxerror]

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


from keras import optimizers

def getOptimizerSGD(learning_rate=0.01, momentum=0.0, nesterov=False):
    return optimizers.SGD(learning_rate=learning_rate, momentum=learning_rate, nesterov=nesterov)

def getOptimizerRMSprop(learning_rate=0.0001, rho=0.9):
    return optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

def getOptimizerAdagrad(learning_rate=0.01):
    return optimizers.Adagrad(learning_rate=learning_rate)

def getOptimizerAdadelta(learning_rate=0.0001, rho=0.95):
    return optimizers.Adadelta(learning_rate=learning_rate, rho=rho)

def getOptimizerAdam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False):
    return optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

def getOptimizerAdamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999):
    return optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

def getOptimizerNadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999):
    return optimizers.Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
