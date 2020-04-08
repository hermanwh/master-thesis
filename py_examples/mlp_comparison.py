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

mlp_1x_16 = mlApi.MLP('mlp 1x16', layers=[16], dropout=0.3)
mlp_1x_32 = mlApi.MLP('mlp 1x32', layers=[32], dropout=0.3)
mlp_1x_64 = mlApi.MLP('mlp 1x64', layers=[64], dropout=0.3)
mlp_1x_128 = mlApi.MLP('mlp 1x128', layers=[128], dropout=0.3)

mlp_2x_16 = mlApi.MLP('mlp 2x16', layers=[16, 16], dropout=0.3)
mlp_2x_32 = mlApi.MLP('mlp 2x32', layers=[32, 32], dropout=0.3)
mlp_2x_64 = mlApi.MLP('mlp 2x64', layers=[64, 64], dropout=0.3)
mlp_2x_128 = mlApi.MLP('mlp 2x128', layers=[128, 128], dropout=0.3)

linear_cv = mlApi.Linear_Regularized('linear')

ensemble = mlApi.Ensemble('mlp 1x64 + linear', [mlp_1x_64, linear_cv])
ensemble2 = mlApi.Ensemble('mlp 2x64 + linear', [mlp_2x_64, linear_cv])

modelList = [
    #mlp_1x_16,
    #mlp_1x_32,
    #mlp_1x_64,
    mlp_1x_128,
    #mlp_2x_16,
    #mlp_2x_32,
    mlp_2x_64,
    mlp_2x_128,
    ensemble,
    ensemble2,
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