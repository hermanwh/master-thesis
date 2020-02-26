import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.dates as mdates
import pyims
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error

INTERPOLDEG = 3

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
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    if columnDescriptions:
        ax.set_ylabel(columnDescriptions[column])
        ax.set_title("Deviation for " + columnDescriptions[column])
    else:
        ax.set_ylabel(column)
        ax.set_title("Deviation for " + column)
    ax.plot(df.index, data, color=color, label="Data")
    ax.axvline(x=pd.to_datetime("2018-05-01 00:00:00", dayfirst=True), color='blue')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(1, axis='y')

    z = np.polyfit(range(len(data)), data, INTERPOLDEG)
    p = np.poly1d(z)
    func = p(range(len(data)))
    ax.plot(df.index, func, color='black', label="Pol.fit")

    fig.subplots_adjust(right=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.autofmt_xdate()

def plotDataColumn(df, plt, column, pred, y, columnDescriptions=None, color1='darkgreen', color2='red'):
    fig, ax = plt.subplots()
    color = color1
    ax.set_xlabel('Date')
    if columnDescriptions:
        ax.set_ylabel(columnDescriptions[column])
        ax.set_title("Predictions for " + columnDescriptions[column])
    else:
        ax.set_ylabel(column)
        ax.set_title("Predictions for " + column)
    ax.plot(df.index, pred, color=color, label="Prediction", alpha=0.5)
    z = np.polyfit(range(len(pred)), pred, INTERPOLDEG)
    p = np.poly1d(z)
    func = p(range(len(pred)))
    ax.plot(df.index, func, color=color, label="Pol. fit, pred")

    ax.axvline(x=pd.to_datetime("2018-05-01 00:00:00", dayfirst=True), color='blue')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(1, axis='y')

    color = color2
    ax.plot(df.index, y, color=color, label="Target", alpha=0.5)
    z = np.polyfit(range(len(y)), y, INTERPOLDEG)
    p = np.poly1d(z)
    func = p(range(len(y)))
    ax.plot(df.index, func, color=color, label="Pol. fit, target")

    fig.subplots_adjust(right=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.autofmt_xdate()

def plotData(df, plt, columnDescriptions=None, relevantColumns=None, color='darkgreen'):
    if relevantColumns is not None:
        columns = relevantColumns
    else:
        columns = df.columns

    columnDescKeys = list(columnDescriptions.keys())
    dfcolumns = df.columns

    for column in columns:
        if column != "Date":
            if  column in df.columns:
                fig, ax = plt.subplots()
                ax.set_title('Plot of dataset column ' + column)
                ax.set_xlabel('Date')
                print(column)
                if columnDescriptions is not None and column in columnDescKeys:
                    ax.set_ylabel(column + " " + columnDescriptions[column], color=color)
                else:
                    ax.set_ylabel(column, color=color)
                ax.plot(df.index, df[column], color=color)
                ax.tick_params(axis='y', labelcolor=color)
                ax.grid(1, axis='y')
            else:
                print("Column " + column + "not in dataset")

def plotDataByTimeframe(df, plt, start, end, columnDescriptions=None):
    df = getDataByTimeframe(df, start, end)
    for column in df.columns:
        if column != "Date":
            fig, ax = plt.subplots()
            ax.set_title('Plot of dataset column ' + column)
            color = 'darkgreen'
            ax.set_xlabel('Date')
            ax.set_ylabel(columnDescriptions[column], color=color)
            ax.plot(df.index, df[column], color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(1, axis='y')
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
    if y_true.shape[1] > 1:
        maxerror = None
    else:
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
