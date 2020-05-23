# Uncertainty of Recurrent Neural Network models with dropout at time of prediction

# %load 4_pred_uncertainty.py
import src.core as mlModule
import src.core_configs as configs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotDropoutPrediction(modelList, predictions, means, stds, targetColumns, df_test, y_test, traintime=None):
    if traintime is not None:
        trainEndStr = [item for sublist in traintime for item in sublist]
    else:
        trainEndStr = None

    for i in range(len(modelList)):
        output_mean = means[i]
        output_std = stds[i]

        for j in range(output_mean.shape[-1]):
            mean = output_mean[:, j]
            std = output_std[:, j]

            upper = np.add(mean, std)
            lower = np.subtract(mean, std)

            fig, ax = plt.subplots(1, 1, figsize=(10,3), dpi=100)
            ax.set_xlabel('Date')
            ax.set_ylabel(mlModule._columnUnits[targetColumns[j]])
            ax.set_title(modelList[i].name + "\nPredictions and targets, " + mlModule._columnDescriptions[targetColumns[j]])
            ax.plot(df_test.iloc[mlModule._maxEnrolWindow:].index, y_test[mlModule._maxEnrolWindow:, j], color="red", alpha=0.5, label="targets")
            ax.plot(df_test.iloc[mlModule._maxEnrolWindow:].index, upper, color="grey", alpha=0.7, label="+/- 1 std bounds")
            ax.plot(df_test.iloc[mlModule._maxEnrolWindow:].index, lower, color="grey", alpha=0.7)
            ax.plot(df_test.iloc[mlModule._maxEnrolWindow:].index, mean, color="blue", alpha=1.0, label="prediction")
            ax.grid(1, axis='y')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
            
            if trainEndStr:
                for i, trainEndString in enumerate(trainEndStr):
                    ax.axvline(x=pd.to_datetime(trainEndString, dayfirst=True), color='black' if i % 2 == 0 else 'blue', label='start training' if i % 2 == 0 else 'end training')

            plt.show()

def performDropoutPrediction(facility, model, resolution, lookback=12, retrain=False):
    filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(facility, model, resolution)

    df = mlModule.initDataframe(filename, columns, irrelevantColumns)
    df_train, df_test = mlModule.getTestTrainSplit(traintime, testtime)
    X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(targetColumns)

    lstm = mlModule.LSTM('LSTMs 1x128 d0.2 mod'+model, layers=[128], training=True, dropout=0.2, recurrentDropout=0.2, enrolWindow=lookback)
    gru = mlModule.GRU('GRUs 1x128 d0.2 mod'+model, layers=[128], training=True, dropout=0.2, recurrentDropout=0.2, enrolWindow=lookback)
    
    modelList = [
        lstm,
        gru,
    ]

    mlModule.initModels(modelList)
    mlModule.trainModels(retrain)

    predictions, means, stds = mlModule.predictWithModelsUsingDropout(numberOfPredictions=30)
    plotDropoutPrediction(modelList, predictions, means, stds, targetColumns, df_test, y_test, traintime)

model = 'A'

performDropoutPrediction('F', model, '30min', 12, retrain=False)
mlModule.reset()
performDropoutPrediction('G', model, '30min', 12, retrain=False)
mlModule.reset()
performDropoutPrediction('G', model, '10min', 12*3, retrain=False)
