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

    lstm_1x_16 = mlModule.LSTM('LSTM 1x16'+' mod'+model, layers=[16], dropout=0.2, recurrentDropout=0.2, epochs=1000)
    lstm_1x_32 = mlModule.LSTM('LSTM 1x32'+' mod'+model, layers=[32], dropout=0.2, recurrentDropout=0.2, epochs=1000)
    lstm_1x_64 = mlModule.LSTM('LSTM 1x64'+' mod'+model, layers=[64], dropout=0.2, recurrentDropout=0.2, epochs=1000)
    lstm_1x_128 = mlModule.LSTM('LSTM 1x128'+' mod'+model, layers=[128], dropout=0.2, recurrentDropout=0.2, epochs=1000)

    lstm_2x_16 = mlModule.LSTM('LSTM 2x16'+' mod'+model, layers=[16, 16], dropout=0.2, recurrentDropout=0.2, epochs=1000)
    lstm_2x_32 = mlModule.LSTM('LSTM 2x32'+' mod'+model, layers=[32, 32], dropout=0.2, recurrentDropout=0.2, epochs=1000)
    lstm_2x_64 = mlModule.LSTM('LSTM 2x64'+' mod'+model, layers=[64, 64], dropout=0.2, recurrentDropout=0.2, epochs=1000)
    lstm_2x_128 = mlModule.LSTM('LSTM 2x128'+' mod'+model, layers=[128, 128], dropout=0.2, recurrentDropout=0.2, epochs=1000)

    linear_cv = mlModule.Linear_Regularized('Linear rCV'+' mod'+model)

    ensemble = mlModule.Ensemble('LSTM 1x128 + Linear'+' mod'+model, [lstm_1x_128, linear_cv])
    ensemble2 = mlModule.Ensemble('LSTM 2x64 + Linear'+' mod'+model, [lstm_2x_64, linear_cv])

    modelList = [
        linear_cv,
        lstm_1x_16,
        lstm_1x_32,
        lstm_2x_16,
        lstm_2x_32,
    ]

    initTrainPredict(modelList)

    modelList = [
        linear_cv,
        lstm_1x_64,
        lstm_1x_128,
        lstm_2x_64,
        lstm_2x_128,
    ]

    initTrainPredict(modelList)

    modelList = [
        linear_cv,
        ensemble,
        ensemble2,
    ]

    initTrainPredict(modelList)

pred('F', 'A', '30min')
mlModule.reset()
pred('F', 'B', '30min')
mlModule.reset()
pred('G', 'A', '30min')
mlModule.reset()
pred('G', 'B', '30min')
mlModule.reset()
