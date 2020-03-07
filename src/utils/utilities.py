import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.dates as mdates
import pyims
from keras import optimizers
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

def plotColumns(df, plt, args, desc="", columnDescriptions=None, trainEndStr=None, columnUnits=None, interpol=False):
    fig,ax = plt.subplots()
    ax.set_xlabel('Date')
    for i, arg in enumerate(args):
        label, column, data, color, alpha = arg
        
        ax.set_title((desc + columnDescriptions[column]) if columnDescriptions else (desc + column))
        ax.set_ylabel(columnUnits[column] if columnUnits is not None else "")

        if color is not None:
            ax.plot(df.index, data, color=color, label=label, alpha=alpha)
        else:
            ax.plot(df.index, data, label=label, alpha=alpha)
    if interpol:
        for i, arg in enumerate(args):
            label, column, data, color, alpha = arg
            z = np.polyfit(range(len(data)), data, INTERPOLDEG)
            p = np.poly1d(z)
            func = p(range(len(data)))
            if color is not None:
                ax.plot(df.index, func, color=color, label="Pol. fit, " + label, alpha=1.0)
            else:
                ax.plot(df.index, func, label="Pol. fit, " + label, alpha=1.0)

    if trainEndStr:
        ax.axvline(x=pd.to_datetime(trainEndStr, dayfirst=True), color='blue')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(1, axis='y')

    fig.subplots_adjust(right=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.autofmt_xdate()

def duoPlot(y1, y2, x, plt, columnDescriptions=None, relevantColumns=None, columnUnits=None, color1='darkgreen', color2='red'):
    fig, ax1 = plt.subplots(1, 1, figsize=(8,6), dpi=100)
    ax1.plot(x, y1, color=color1, alpha=1.0)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2, alpha=0.5)

    ax1.set_xlabel('x axis', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Label 1', color=color1, fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
    ax2.set_ylabel("Label 2", color=color2, fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax1.grid(alpha=.4)
    ax2.set_title("Plot", fontsize=22)
    plt.show()

def plotData(df, plt, columnDescriptions=None, relevantColumns=None, columnUnits=None, color='darkgreen'):
    if relevantColumns is not None:
        columns = relevantColumns
    else:
        columns = df.columns

    columnDescKeys = list(columnDescriptions.keys())
    columnUnitKeys = list(columnUnits.keys())
    dfcolumns = df.columns

    #duoPlot(df['TT0102_MA_Y'], df['TT0106_MA_Y'], df.index, plt, columnDescriptions=columnDescriptions, relevantColumns=relevantColumns, columnUnits=columnUnits)

    for column in columns:
        if column != "Date":
            if  column in df.columns:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
                ax.set_xlabel('Date')
                if columnDescriptions is not None and column in columnDescKeys:
                    ax.set_title(columnDescriptions[column] + " " + column)
                else:
                    ax.set_title(column)
                if columnUnits is not None and column in columnUnitKeys:
                    ax.set_ylabel(columnUnits[column])
                else:
                    ax.set_ylabel(column)
                plt.gca().spines["top"].set_alpha(.3)
                plt.gca().spines["bottom"].set_alpha(.3)
                plt.gca().spines["right"].set_alpha(.3)
                plt.gca().spines["left"].set_alpha(.3)
                plt.plot(df.index, df[column], label=column, lw=1.5, color=color)
                plt.text(df.shape[0]+1, df[column].values[-1], column, fontsize=14, color=color)
                plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
                plt.grid(1, alpha=0.5)
                plt.legend(loc=(1.01, 0.01), ncol=1)
            else:
                print("Column " + column + "not in dataset")

    

def plotDataByTimeframe(df, plt, start, end, columnDescriptions=None, relevantColumns=None):
    df = getDataByTimeframe(df, start, end)
    plotData(df, plt, columnDescriptions=columnDescriptions, relevantColumns=relevantColumns)

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

def saveKerasModel(model, loc, name):
    print("Saving model")
    model.save(loc + '/' + name + '.h5')
    print("Model saved")
    printHorizontalLine()

def calculateR2Score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

def calculateMetrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    if len(y_true.shape) > 1 and y_true.shape and y_true.shape[1] > 1:
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
