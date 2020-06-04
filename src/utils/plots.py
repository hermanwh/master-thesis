import numpy as np
import pandas as pd
import matplotlib.pyplot as pltt

np.random.seed(100)

def getPlotColors():
    colors = [
        '#000080',
        '#2ca25f',
        '#8856a7',
        '#43a2ca',
        '#e34a33',
        '#636363',
        '#663300',
        '#003300',
        '#ff3399',
        '#99d8c9',
        '#9ebcda',
        '#fdbb84',
        '#c994c7',
        'darkgreen',
        'darkred',
        'darkgrey',
    ]
    
    return colors

def plotDataColumnSingle(dfindex, plt, column, data, columnDescriptions=None, color='darkgreen', interpoldeg=3):
    # Plots a single data column based on index, plt object, columnName and data

    fig, ax = plt.subplots(1, 1, figsize=(10,3), dpi=100)
    ax.set_xlabel('Date')
    if columnDescriptions:
        ax.set_ylabel(columnDescriptions[column])
        ax.set_title("Deviation for " + columnDescriptions[column])
    else:
        ax.set_ylabel(column)
        ax.set_title("Deviation for " + column)
    ax.plot(dfindex, data, color=color, label="Data")
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(1, axis='y')

    z = np.polyfit(range(len(data)), data, interpoldeg)
    p = np.poly1d(z)
    func = p(range(len(data)))
    ax.plot(dfindex, func, color='black', label="Pol.fit")

    fig.subplots_adjust(right=0.7)
 
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    fig.autofmt_xdate()

def plotColumns(
        dfindex,
        plt,
        args,
        desc="",
        columnDescriptions=None,
        trainEndStr=None,
        columnUnits=None,
        alpha=0.8,
        interpol=False,
        interpoldeg=3,
    ):
    # Primary plotting method
    # Plots provided data from a predefined args format:
    # [label, column, data, color]

    fig,ax = plt.subplots(1, 1, figsize=(10,3), dpi=100)
    ax.set_xlabel('Date')
    for i, arg in enumerate(args):
        label, column, data, color = arg
        
        ax.set_title((desc + "\n" + columnDescriptions[column]) if columnDescriptions else (desc + "\n" + column))
        ax.set_ylabel(columnUnits[column] if columnUnits is not None else "")

        if color is not None:
            ax.plot(dfindex, data, color=color, label=label, alpha=alpha)
        else:
            ax.plot(dfindex, data, label=label, alpha=alpha)
    if interpol:
        for i, arg in enumerate(args):
            label, column, data, color, alpha = arg
            z = np.polyfit(range(len(data)), data, interpoldeg)
            p = np.poly1d(z)
            func = p(range(len(data)))
            if color is not None:
                ax.plot(dfindex, func, color=color, label="Pol. fit, " + label, alpha=1.0)
            else:
                ax.plot(dfindex, func, label="Pol. fit, " + label, alpha=1.0)

    if trainEndStr:
        for i, trainEndString in enumerate(trainEndStr):
            ax.axvline(x=pd.to_datetime(trainEndString, dayfirst=True), color='black' if i % 2 == 0 else 'blue', label='start training' if i % 2 == 0 else 'end training')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(1, axis='y')

    fig.subplots_adjust(right=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    fig.autofmt_xdate()

def duoPlot(y1, y2, x, plt, columnDescriptions=None, relevantColumns=None, columnUnits=None, color1=getPlotColors()[0], color2=getPlotColors()[1], y2lim=None, textArgs=['Duo plot', 'Date', 'Label 1', 'Label 2']):
    # Plots two columns in the same figure with two different axes

    title, xaxis, y1axis, y2axis = textArgs

    fig, ax1 = plt.subplots(1, 1, figsize=(10,3), dpi=100)
    ax1.plot(x, y1, color=color1, alpha=0.8)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2, alpha=0.8)

    if y2lim is not  None:
        (ymin, ymax) = y2lim
        ax2.set_ylim([ymin, ymax])

    ax1.set_xlabel(xaxis)
    ax1.tick_params(axis='x', rotation=0)
    ax1.set_ylabel(y1axis, color=color1)
    ax1.tick_params(axis='y', rotation=0)
    ax2.set_ylabel(y2axis, color=color2)
    ax2.tick_params(axis='y')

    ax1.grid(alpha=.4)
    ax2.set_title(title)
    plt.show()

def plotTraining(history, plt):
    # Plots the training history of a set of models

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    plt.plot(history.history['mean_squared_error'], color='blue', label="Training loss", alpha=0.8)
    plt.plot(history.history['val_mean_squared_error'], color="orange", label="Validation loss", alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.title('Training history')
    plt.show()

def plotData(df, plt, columnDescriptions={}, relevantColumns=None, columnUnits=None, color=getPlotColors()[0]):
    # Plots the columns of a pandas dataframe

    if relevantColumns is not None:
        columns = relevantColumns
    else:
        columns = df.columns

    columnDescKeys = list(columnDescriptions.keys())
    columnUnitKeys = list(columnUnits.keys()) if columnUnits is not None else []
    dfcolumns = df.columns

    for column in columns:
        if column != "Date":
            if  column in df.columns:
                fig, ax = plt.subplots()
                ax.set_xlabel('Date')
                if columnDescriptions is not None and column in columnDescKeys:
                    ax.set_title(columnDescriptions[column] + " " + column)
                else:
                    ax.set_title(column)
                if columnUnits is not None and column in columnUnitKeys:
                    ax.set_ylabel(columnUnits[column])
                else:
                    ax.set_ylabel(column)
                #plt.gca().spines["top"].set_alpha(.3)
                #plt.gca().spines["bottom"].set_alpha(.3)
                #plt.gca().spines["right"].set_alpha(.3)
                #plt.gca().spines["left"].set_alpha(.3)
                plt.plot(df.index, df[column], label=column, lw=1.5, color=color, alpha=0.8)
                #plt.text(df.shape[0]+1, df[column].values[-1], column, fontsize=14, color=color)
                #plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
                plt.grid(1, alpha=0.5)
                plt.legend(loc=(1.01, 0.01), ncol=1)
            else:
                print("Column " + column + " not in dataset")
    plt.show()

def plotDataByTimeframe(df, plt, start, end, columnDescriptions=None, relevantColumns=None):
    # Plots the columns of a pandas dataframe based on provided timeframe
    
    df = getDataByTimeframe(df, start, end)
    plotData(df, plt, columnDescriptions=columnDescriptions, relevantColumns=relevantColumns)

def plotModelScores(plt, names, r2_train, r2_test, test=False):
    # Plots calculated scores (training + potentially test) of a machine learning model

    fig, ax = plt.subplots(1, 1, figsize=(10,3), dpi=100)
    ax.set_ylabel('R2 score')
    ax.set_xlabel('Model')
    ax.set_title('Model metrics')

    ax.plot(names, r2_train, marker='x', markersize=10, label="Training metrics")
    if test:
        ax.plot(names, r2_test, marker='x', markersize=10, label="Test metrics")
    ax.legend()

    plt.show()

def plotModelPredictions(plt, deviationsList, columnsList, indexList, labelNames, columnUnits, traintime, interpol=False, interpoldeg=10):
    # Primary plotting entry point used in low-level API
    # deviationsList is a set of deviations between predicted and actual value for all outputs for all models
    # columnsList is a set of predictions for all outputs for all models

    for i in range(len(deviationsList)):
        plotColumns(
            indexList,
            plt,
            columnsList[i],
            desc="Prediction and targets",
            columnDescriptions=labelNames,
            columnUnits=columnUnits,
            trainEndStr=[item for sublist in traintime for item in sublist],
            interpol=interpol,
            interpoldeg=interpoldeg,
        )
        plotColumns(
            indexList,
            plt,
            deviationsList[i],
            desc="Deviation",
            columnDescriptions=labelNames,
            columnUnits=columnUnits,
            trainEndStr=[item for sublist in traintime for item in sublist],
            interpol=interpol,
            interpoldeg=interpoldeg,
        )

    plt.show()

def plotTrainingSummary(trainingSummary):
    # Plots the training history generated by the modelFuncs.getTrainingSummary() method

    colors = getPlotColors()

    fig,axs = pltt.subplots(nrows=1, ncols=2, figsize=(10, 3), dpi=100)
    fig.tight_layout(w_pad=3.0)

    ax1, ax2 = axs

    max_y = 5.0 *  np.mean(list(map(lambda x: x['loss_actual'], trainingSummary.values())))
    max_yval = 5.0 * np.mean(list(map(lambda x: x['val_loss_final'], trainingSummary.values())))

    ax1.set_ylim([0, max_y])
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.set_ylim([0, max_yval])
    ax2.set_title('Validation loss')
    ax2.set_ylabel('Val. loss')
    ax2.set_xlabel('Epoch')

    for i, (name, summary) in enumerate(trainingSummary.items()):
        ax1.plot(summary['loss'], color=colors[i], label=name, alpha=0.8)
        ax2.plot(summary['val_loss'], color=colors[i], label=name, alpha=0.8)
        #ax1.plot(summary['loss_loc'], summary['loss_final'], color=colors[i], marker='o', markersize=9, label="Min. loss", alpha=0.7)
        ax1.plot(summary['val_loss_loc'], summary['loss_actual'], color=colors[i], marker='x', markersize=10, label="Chosen loss")
        ax2.plot(summary['val_loss_loc'], summary['val_loss_final'], color=colors[i], marker='x', markersize=10, label="Min. val loss")

    ax1.legend(loc='upper right', prop={'size': 10})
    ax2.legend(loc='upper right', prop={'size': 10})
    pltt.show()