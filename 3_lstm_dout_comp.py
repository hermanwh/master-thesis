# LSTM architecture comparison

# %load lstm_comparison.py
import src.core as mlModule
import src.core_configs as configs

def initTrainPredict(modelList, retrain=False, plot=True, score=True):
    mlModule.initModels(modelList)
    mlModule.trainModels(retrain)
    
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
        plot=plot,
        score=score,
    )
    

def pred(facility, model, resolution):
    filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(facility, model, resolution)

    df = mlModule.initDataframe(filename, columns, irrelevantColumns)
    df_train, df_test = mlModule.getTestTrainSplit(traintime, testtime)
    X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(targetColumns)

    lstm_1_1 = mlModule.LSTM('LSTM 1x128 d0.0'+' mod'+model, layers=[128], dropout=0.0, recurrentDropout=0.0, epochs=5000)
    lstm_1_2 = mlModule.LSTM('LSTM 1x128 d0.1'+' mod'+model, layers=[128], dropout=0.1, recurrentDropout=0.1, epochs=5000)
    lstm_1_3 = mlModule.LSTM('LSTM 1x128 d0.2'+' mod'+model, layers=[128], dropout=0.2, recurrentDropout=0.2, epochs=5000)
    lstm_1_4 = mlModule.LSTM('LSTM 1x128 d0.3'+' mod'+model, layers=[128], dropout=0.3, recurrentDropout=0.3, epochs=5000)
    lstm_1_5 = mlModule.LSTM('LSTM 1x128 d0.4'+' mod'+model, layers=[128], dropout=0.4, recurrentDropout=0.4, epochs=5000)
    lstm_1_6 = mlModule.LSTM('LSTM 1x128 d0.5'+' mod'+model, layers=[128], dropout=0.5, recurrentDropout=0.5, epochs=5000)
    linear = mlModule.Linear_Regularized('Linear rCV mod'+model)

    initTrainPredict([
        linear, lstm_1_1, lstm_1_2, lstm_1_3, lstm_1_4, lstm_1_5, lstm_1_6,
    ])

pred('G', 'A', '30min')
mlModule.reset()
pred('G', 'A', '10min')
mlModule.reset()