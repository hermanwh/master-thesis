import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error
from configs import getConfig
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau
import metrics
from keras.utils import plot_model
import plots
import ast

INTERPOLDEG = 3

class Config():
    def __init__(self, config):
        self.columns = config['columns']
        self.relevantColumns = config['relevantColumns']
        self.labelNames = config['labelNames']
        self.columnUnits = config['columnUnits']
        self.timestamps = config['timestamps']

class Args():
    def __init__(self, args):
        self.activation = args['activation']
        self.loss = args['loss']
        self.optimizer = args['optimizer']
        self.metrics = args['metrics']
        self.epochs = args['epochs']
        self.batchSize = args['batchSize']
        self.verbose = args['verbose']
        self.callbacks= args['callbacks']
        self.enrolWindow = args['enrolWindow']
        self.validationSize = args['validationSize']
        self.testSize = args['testSize']

def getBasicCallbacks(monitor="val_loss", patience_es=200, patience_rlr=80):
    return [
        EarlyStopping(
            monitor = monitor, min_delta = 0.00001, patience = patience_es, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = monitor, factor = 0.5, patience = patience_rlr, verbose = 1, min_lr=5e-4,
        )
    ]

def getBasicHyperparams():
    return {
        'activation': 'relu',
        'loss': 'mean_squared_error',
        'optimizer': 'adam',
        'metrics': ['mean_squared_error'],
    }

def getFeatureTargetSplit(df_train, df_test, targetColumns):
    X_train = df_train.drop(targetColumns, axis=1).values
    y_train = df_train[targetColumns].values

    X_test = df_test.drop(targetColumns, axis=1).values
    y_test = df_test[targetColumns].values

    return [X_train, y_train, X_test, y_test]

def getTestTrainSplit(df, traintime, testtime):
    start_train, end_train = traintime[0]
    df_train = getDataByTimeframe(df, start_train, end_train)
    for start_train, end_train in traintime[1:]:
        nextDf = getDataByTimeframe(df, start_train, end_train)
        df_train = pd.concat([df_train, nextDf])

    start_test, end_test = testtime
    df_test = getDataByTimeframe(df, start_test, end_test)

    return [df_train, df_test]

def printModelSummary(model):
    if hasattr(model, "summary"):
        # Keras Model object
        print(model.summary())
    elif hasattr(model, "model"):
        # MachineLearningModel object
        if hasattr(model.model, "summary"):
            # MachineLearningModel.model will be a Keras Model object
            printModelSummary(model.model)
    elif hasattr(model, "models"):
        # EnsembleModel object
        print("Model is of type Ensemble Model")
        print("Sub model summaries will follow")
        print("-------------------------------")
        for mod in model.models:
            # EnsembleModel.models will be a list of MachineLearningModels
            printModelSummary(mod)
    else:
        print("Simple models have no summary")
    
def printModelWeights(model):
    if hasattr(model, "summary"):
        # Keras Model object
        for layer in model.layers: print(layer.get_config(), layer.get_weights())
    elif hasattr(model, "model"):
        # MachineLearningModel object
        if hasattr(model.model, "summary"):
            # MachineLearningModel.model will be a Keras Model object
            printModelSummary(model.model)
    elif hasattr(model, "models"):
        # EnsembleModel object
        print("Model is of type Ensemble Model")
        print("Sub model summaries will follow")
        print("-------------------------------")
        for mod in model.models:
            # EnsembleModel.models will be a list of MachineLearningModels
            printModelSummary(mod)
    else:
        if hasattr(model, "get_params"):
            print(model.get_params())
        else:
            print("No weights found")

def plotKerasModel(model):
    plot_model(model.model)

def printModelPredictions(names, r2_train, r2_test):
    plt.ylabel('Stuff')
    plt.title('Things')

    plt.plot(names, r2_train)
    plt.plot(names, r2_test)

    plt.show()

def plotModelPredictions(plt, deviationsList, columnsList, df_test, labelNames, traintime):
    for i in range(len(deviationsList)):
        plots.plotColumns(
            df_test,
            plt,
            deviationsList[i],
            desc="Deviation, ",
            columnDescriptions=labelNames,
            trainEndStr=[item for sublist in traintime for item in sublist],
            interpol=False,
        )
        plots.plotColumns(
            df_test,
            plt,
            columnsList[i],
            desc="Prediction vs. targets, ",
            columnDescriptions=labelNames,
            trainEndStr=[item for sublist in traintime for item in sublist],
            interpol=False,
        )

    plt.show()

def predictWithModel(model, X_train, y_train, X_test, y_test, targetColumns):
    return predictWithModels([model], X_train, y_train, X_test, y_test, targetColumns)

def predictWithModels(modelsList, X_train, y_train, X_test, y_test, targetColumns):
    colors = getPlotColors()
    
    names = []
    r2_train = []
    r2_test = []

    deviationsList = []
    columnsList = []
    for i in range(y_train.shape[1]):
        columnsList.append([])
        columnsList[i].append([
                        'Targets',
                        targetColumns[i],
                        y_test[:, i],
                        'red',
                        0.5,
                    ])

        deviationsList.append([])

    for i, modObj in enumerate(modelsList):
        mod, name = modObj
        
        mod.train()
        if mod.args.enrolWindow is not None:
            pred_train = mod.predict(X_train, y=y_train)
            pred_test = mod.predict(X_test, y=y_test)
        else:
            pred_train = mod.predict(X_train, y=None)
            pred_test = mod.predict(X_test, y=None)

        if mod.args.enrolWindow is not None:
            train_metrics = metrics.calculateMetrics(y_train[mod.args.enrolWindow:], pred_train)
            test_metrics = metrics.calculateMetrics(y_test[mod.args.enrolWindow:], pred_test)
        else:
            train_metrics = metrics.calculateMetrics(y_train, pred_train)
            test_metrics = metrics.calculateMetrics(y_test, pred_test)
        
        for j in range(y_train.shape[1]):
            columnsList[j].append(
                [
                    name,
                    targetColumns[j],
                    pred_test[:, j],
                    colors[i],
                    1.0,
                ]
            )
            deviationsList[j].append(
                [
                    name,
                    targetColumns[j],
                    y_test[:, j] - pred_test[:, j],
                    colors[i],
                    1.0,
                ]
            )

        r2_train.append(train_metrics[0])
        r2_test.append(test_metrics[0])
        names.append(name)   
    
    return [
        names,
        r2_train,
        r2_test,
        deviationsList,
        columnsList,
    ]

def getPlotColors():
    #colors = ['#92a8d1','#034f84','#f7cac9','#f7786b','#deeaee','#b1cbbb','#eea29a','#c94c4c']
    colors = ['#686256','#c1502e','#587e76','#a96e5b','#454140','#bd5734','#7a3b2e']
    """
    colors = [
        '#0C0910',
        '#453750',
        '#73648A',
        '#9882AC',
        '#A393BF',
        '#8AAA79',
        '#657153',
        '#837569',
        '#B7B6C2',
        '#D1D5DE',
        '#D58936',
        '#A44200',
        '#69140E',
        '#3C1518'
    ]
    """
    return colors

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
