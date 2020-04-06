import src.core as mlApi
import src.core_configs as configs

import numpy as np
import matplotlib.pyplot as plt

def plotDropoutPrediction(modelList, predictions, means, stds):
	global df_test

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
			ax.set_ylabel(mlApi._columnUnits[targetColumns[j]])
			ax.set_title(modelList[i].name + "\nPredictions and targets, " + mlApi._columnDescriptions[targetColumns[j]])
			ax.plot(df_test.iloc[mlApi._maxEnrolWindow:].index, y_test[mlApi._maxEnrolWindow:, j], color="red", alpha=0.5, label="targets")
			ax.plot(df_test.iloc[mlApi._maxEnrolWindow:].index, upper, color="grey", alpha=0.7, label="+/- 1 std bounds")
			ax.plot(df_test.iloc[mlApi._maxEnrolWindow:].index, lower, color="grey", alpha=0.7)
			ax.plot(df_test.iloc[mlApi._maxEnrolWindow:].index, mean, color="blue", alpha=1.0, label="prediction")
			ax.grid(1, axis='y')
			ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
			
			plt.show()

# 1. Define dataset specifics
filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('F', 'B', '30min')

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

# 3. Define models | NB: only RNN (LSTM/GRU) models
lstmd_1x_128 = mlApi.LSTM('lstmd 1x 128 training', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)
lstmd_2x_64 = mlApi.LSTM('lstmd 2x 64 training', layers=[64, 64], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)
gru_1x_128 = mlApi.GRU('grud 1x 128 training', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)

modelList = [
	lstmd_1x_128,
	lstmd_2x_64,
	gru_1x_128,
]

# 4. Initiate and train models
retrain=False
mlApi.initModels(modelList)
mlApi.trainModels(retrain)

# 5. Predict
predictions, means, stds = mlApi.predictWithModelsUsingDropout(numberOfPredictions=30)

plotDropoutPrediction(modelList, predictions, means, stds)

mlApi.reset()

# 1. Define dataset specifics
filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('G', 'A', '30min')

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

# 3. Define models | NB: only RNN (LSTM/GRU) models
lstmd_1x_128 = mlApi.LSTM('lstmd 1x 128 training', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)
lstmd_2x_64 = mlApi.LSTM('lstmd 2x 64 training', layers=[64, 64], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)
gru_1x_128 = mlApi.GRU('grud 1x 128 training', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)

modelList = [
	lstmd_1x_128,
	lstmd_2x_64,
	gru_1x_128,
]

# 4. Initiate and train models
retrain=False
mlApi.initModels(modelList)
mlApi.trainModels(retrain)

# 5. Predict
predictions, means, stds = mlApi.predictWithModelsUsingDropout(numberOfPredictions=30)

plotDropoutPrediction(modelList, predictions, means, stds)

mlApi.reset()

# 1. Define dataset specifics
filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('G', 'C', '10min')

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

# 3. Define models | NB: only RNN (LSTM/GRU) models
lstmd_1x_128 = mlApi.LSTM('lstmd 1x 128 training', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)
lstmd_2x_64 = mlApi.LSTM('lstmd 2x 64 training', layers=[64, 64], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)
gru_1x_128 = mlApi.GRU('grud 1x 128 training', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500, enrolWindow=12)

modelList = [
	lstmd_1x_128,
	lstmd_2x_64,
	gru_1x_128,
]

# 4. Initiate and train models
retrain=False
mlApi.initModels(modelList)
mlApi.trainModels(retrain)

# 5. Predict
predictions, means, stds = mlApi.predictWithModelsUsingDropout(numberOfPredictions=30)

plotDropoutPrediction(modelList, predictions, means, stds)