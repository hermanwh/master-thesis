import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import src.core as mlApi
import src.core_configs as configs

# 1. Define dataset specifics

filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('G', 'C', '10min')

# 2. Initiate and divide data

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

# 3. Define models

lstm_1x_128 = mlApi.LSTM('lstm 1x 128', layers=[128], enrolWindow=16)

lstmd_1x_16 = mlApi.LSTM('lstmd 1x 16', layers=[16], dropout=0.3, enrolWindow=16)
lstmd_1x_32 = mlApi.LSTM('lstmd 1x 32', layers=[32], dropout=0.3, enrolWindow=16)
lstmd_1x_64 = mlApi.LSTM('lstmd 1x 64', layers=[64], dropout=0.3, enrolWindow=16)
lstmd_1x_128 = mlApi.LSTM('lstmd 1x 128', layers=[128], dropout=0.3, enrolWindow=16)

lstmd_2x_16 = mlApi.LSTM('lstmd 2x 16', layers=[16, 16], dropout=0.3, enrolWindow=16)
lstmd_2x_32 = mlApi.LSTM('lstmd 2x 32', layers=[32, 32], dropout=0.3, enrolWindow=16)
lstmd_2x_64 = mlApi.LSTM('lstmd 2x 64', layers=[64, 64], dropout=0.3, enrolWindow=16)
lstmd_2x_128 = mlApi.LSTM('lstmd 2x 128', layers=[128, 128], dropout=0.3, enrolWindow=16)

linear_cv = mlApi.Linear_Regularized('linear r')

mlp_d = mlApi.MLP('mlp for ensemble 2x 64', layers=[64, 64], dropout=0.2)

ensemble = mlApi.Ensemble('lstmd + linear', [lstmd_2x_64, linear_cv])
ensemble2 = mlApi.Ensemble('lstmd2 + linear', [lstmd_1x_128, linear_cv])
ensemble3 = mlApi.Ensemble('lstm + mlp', [lstmd_2x_64, mlp_d])

modelList = [
    lstmd_1x_16,
    lstmd_1x_32,
    lstmd_1x_64,
    lstmd_1x_128,
    lstmd_2x_16,
    lstmd_2x_32,
    lstmd_2x_64,
    lstmd_2x_128,
    ensemble,
    ensemble2,
	ensemble3,
    linear_cv,
]

# 4. Initiate and train models

# Define whether to retrain models or not
retrain=False

mlApi.initModels(modelList)
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
)