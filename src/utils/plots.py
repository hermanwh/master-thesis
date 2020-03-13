import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.dates as mdates

INTERPOLDEG = 3

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

def plotColumns(dfindex, plt, args, desc="", columnDescriptions=None, trainEndStr=None, columnUnits=None, interpol=False):
    fig,ax = plt.subplots()
    ax.set_xlabel('Date')
    for i, arg in enumerate(args):
        label, column, data, color, alpha = arg
        
        ax.set_title((desc + columnDescriptions[column]) if columnDescriptions else (desc + column))
        ax.set_ylabel(columnUnits[column] if columnUnits is not None else "")

        if color is not None:
            ax.plot(dfindex, data, color=color, label=label, alpha=alpha)
        else:
            ax.plot(dfindex, data, label=label, alpha=alpha)
    if interpol:
        for i, arg in enumerate(args):
            label, column, data, color, alpha = arg
            z = np.polyfit(range(len(data)), data, INTERPOLDEG)
            p = np.poly1d(z)
            func = p(range(len(data)))
            if color is not None:
                ax.plot(dfindex, func, color=color, label="Pol. fit, " + label, alpha=1.0)
            else:
                ax.plot(dfindex, func, label="Pol. fit, " + label, alpha=1.0)

    if trainEndStr:
        for i, trainEndString in enumerate(trainEndStr):
            ax.axvline(x=pd.to_datetime(trainEndString, dayfirst=True), color='blue' if i % 2 == 0 else 'red')
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

def plotTraining(history, plt):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    plt.plot(history.history['mean_squared_error'], color='blue', label="Training loss")
    plt.plot(history.history['val_mean_squared_error'], color="orange", label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.title('Training history')
    plt.show()

def plotData(df, plt, columnDescriptions=None, relevantColumns=None, columnUnits=None, color='darkgreen'):
    if relevantColumns is not None:
        columns = relevantColumns
    else:
        columns = df.columns

    columnDescKeys = list(columnDescriptions.keys())
    columnUnitKeys = list(columnUnits.keys()) if columnUnits is not None else []
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