# Loss metric comparison

import src.core as mlModule
import src.core_configs as configs

def initTrainPredict(modelList, retrain=False, plot=True, interpol=False):
    # 4. Initiate and train models
    mlModule.initModels(modelList)
    mlModule.trainModels(retrain)
    
    # 5. Predict
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
        plot=plot,
        interpol=interpol,
    )

def pred(facility, model, resolution):
    filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(facility, model, resolution)

    df = mlModule.initDataframe(filename, columns, irrelevantColumns)
    df_train, df_test = mlModule.getTestTrainSplit(traintime, testtime)
    X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(targetColumns)

    mlp_mae = mlModule.MLP('MLP 1x128 d0.2 mae mod'+model, layers=[128], dropout=0.2, loss='mean_absolute_error', metrics=['mean_absolute_error'])
    mlp_mse = mlModule.MLP('MLP 1x128 d0.2 mse mod'+model, layers=[128], dropout=0.2, loss='mean_squared_error', metrics=['mean_squared_error'])
    lstm_mae = mlModule.LSTM('LSTM 1x128 d0.2 mae mod'+model, layers=[128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12, loss='mean_absolute_error', metrics=['mean_absolute_error'])
    lstm_mse = mlModule.LSTM('LSTM 1x128 d0.2 mse mod'+model, layers=[128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    modelList = [
        mlp_mae,
        mlp_mse,
        lstm_mae,
        lstm_mse,
    ]

    initTrainPredict(modelList)

pred('D', 'A', '30min')
mlModule.reset()
pred('D', 'B', '30min')
mlModule.reset()
pred('F', 'A', '30min')
mlModule.reset()
pred('F', 'B', '30min')
mlModule.reset()
pred('G', 'A', '30min')
mlModule.reset()
pred('G', 'B', '30min')
mlModule.reset()
