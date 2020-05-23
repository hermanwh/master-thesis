import src.core as mlApi
import src.core_configs as configs

def train(facility, model, resolution, retrain=False):
    filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(facility, model, resolution)

    df = mlApi.initDataframe(filename, columns, irrelevantColumns)
    df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
    X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

    mlp_1 = mlApi.MLP('MLP 1x64 d0.2 mod'+model, layers=[64], dropout=0.2)
    mlp_2 = mlApi.MLP('MLP 1x128 d0.2 mod'+model, layers=[128], dropout=0.2)
    mlp_3 = mlApi.MLP('MLP 2x64 d0.2 mod'+model, layers=[64, 64], dropout=0.2)
    mlp_4 = mlApi.MLP('MLP 2x128 d0.2 mod'+model, layers=[128, 128], dropout=0.2)

    lstm_1 = mlApi.LSTM('LSTM 1x64 d0.2 mod'+model, layers=[64], dropout=0.2, recurrentDropout=0.2, enrolWindow=12*3)
    lstm_2 = mlApi.LSTM('LSTM 1x128 d0.2 mod'+model, layers=[128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12*3)
    lstm_3 = mlApi.LSTM('LSTM 2x64 d0.2 mod'+model, layers=[64, 64], dropout=0.2, recurrentDropout=0.2, enrolWindow=12*3)
    lstm_4 = mlApi.LSTM('LSTM 2x128 d0.2 mod'+model, layers=[128, 128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12*3)

    modelList = [
        mlp_1,
        mlp_2,
        mlp_3,
        mlp_4,
        lstm_1,
        lstm_2,
        lstm_3,
        lstm_4,
    ]

    mlApi.initModels(modelList)
    mlApi.trainModels(retrain)

train('G', 'A', '10min')
mlApi.reset()
train('G', 'B', '10min')
mlApi.reset()

"""
train('D', 'A', '30min')
mlApi.reset()
train('D', 'B', '30min')
mlApi.reset()
train('F', 'A', '30min')
mlApi.reset()
train('F', 'B', '30min')
mlApi.reset()
train('G', 'A', '30min')
mlApi.reset()
train('G', 'B', '30min')
mlApi.reset()
"""