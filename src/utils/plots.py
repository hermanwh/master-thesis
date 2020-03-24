import numpy as np
import pandas as pd
import matplotlib.pyplot as pltt

np.random.seed(100)

def getPlotColors():
    #colors = ['#92a8d1','#034f84','#f7cac9','#f7786b','#deeaee','#b1cbbb','#eea29a','#c94c4c']
    colors = ['#686256','#c1502e','#587e76','#a96e5b','#454140','#bd5734','#7a3b2e', '#92a8d1','#034f84','#f7cac9','#f7786b','#deeaee','#b1cbbb','#eea29a','#c94c4c']
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

def plotDataColumnSingle(df, plt, column, data, columnDescriptions=None, color='darkgreen', interpoldeg=3):
    fig, ax = plt.subplots(1, 1, figsize=(8,6), dpi=100)
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

    z = np.polyfit(range(len(data)), data, interpoldeg)
    p = np.poly1d(z)
    func = p(range(len(data)))
    ax.plot(df.index, func, color='black', label="Pol.fit")

    fig.subplots_adjust(right=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.autofmt_xdate()

def plotColumns(dfindex, plt, args, desc="", columnDescriptions=None, trainEndStr=None, columnUnits=None, interpol=False, interpoldeg=3):
    fig,ax = plt.subplots(1, 1, figsize=(10,3), dpi=100)
    ax.set_xlabel('Date')
    for i, arg in enumerate(args):
        label, column, data, color, alpha = arg
        
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

def plotModelScores(plt, names, r2_train, r2_test):
    plt.ylabel('R2 score')
    plt.xlabel('Model')
    plt.title('Model metrics')

    plt.plot(names, r2_train, marker='x', markersize=10, label="Training metrics")
    plt.plot(names, r2_test, marker='x', markersize=10, label="Test metrics")
    plt.legend()

    plt.show()

def plotModelPredictions(plt, deviationsList, columnsList, indexList, labelNames, columnUnits, traintime, interpol=False, interpoldeg=3):
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
    colors = getPlotColors()

    fig,axs = pltt.subplots(nrows=1, ncols=2, figsize=(10, 3), dpi=100)
    fig.tight_layout(w_pad=3.0)

    ax1, ax2 = axs

    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.set_title('Validation loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    i = 0
    for name, summary in trainingSummary.items():
        ax1.plot(summary['loss'], color=colors[i], label=name)
        ax1.plot(summary['loss_loc'], summary['loss_final'], color=colors[i], marker='x', markersize=10)
        ax2.plot(summary['val_loss'], color=colors[i], label=name)
        ax2.plot(summary['val_loss_loc'], summary['val_loss_final'], color=colors[i], marker='x', markersize=10)
        i = i + 1

    ax1.legend(loc='upper right', prop={'size': 10})
    ax2.legend(loc='upper right', prop={'size': 10})
    pltt.show()