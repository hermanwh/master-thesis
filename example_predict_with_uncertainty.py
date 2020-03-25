import statApi
from api import Api
mlApi = Api()

# 1. Define dataset spesifics

# File path to dataset .csv
filename = "../master-thesis-db/datasets/F/data2_30min.csv"

# List of columns on form ['name', 'desc', 'unit']
columns = [
	['FYN0111', 'Gasseksport rate', 'MSm^3/d'],
	['FT0111', 'Gasseksport molvekt','g/mole'],
	['TT0102_MA_Y', 'Varm side A temperatur inn', 'degrees'],
	['TIC0101_CA_YX', 'Varm side A temperatur ut', 'degrees'],
	['TT0104_MA_Y', 'Varm side B temperatur inn', 'degrees'],
	['TIC0103_CA_YX', 'Varm side B temperatur ut', 'degrees'],
	['TT0106_MA_Y', 'Varm side C temperatur inn', 'degrees'],
	['TIC0105_CA_YX', 'Varm side C temperatur ut', 'degrees'],
	['TI0115_MA_Y', 'Scrubber temperatur ut', 'degrees'],
	['PDT0108_MA_Y', 'Varm side A trykkfall', 'Bar'],
	['PDT0119_MA_Y', 'Varm side B trykkfall', 'Bar'],
	['PDT0118_MA_Y', 'Varm side C trykkfall', 'Bar'],
	['PIC0104_CA_YX', 'Innløpsseparator trykk', 'Barg'],
	['TIC0425_CA_YX', 'Kald side temperatur inn', 'degrees'],
	['TT0651_MA_Y', 'Kald side A temperatur ut', 'degrees'],
	['TT0652_MA_Y', 'Kald side B temperatur ut', 'degrees'],
	['TT0653_MA_Y', 'Kald side C temperatur ut', 'degrees'],
	['TIC0101_CA_Y', 'Kald side A ventilåpning', '%'],
	['TIC0103_CA_Y', 'Kald side B ventilåpning', '%'],
	['TIC0105_CA_Y', 'Kald side C ventilåpning', '%'],
]

# List of column names to ignore completely
irrelevantColumns = [
		'FT0111',
		'PDT0108_MA_Y',
		'PDT0119_MA_Y',
		'PDT0118_MA_Y',
		'TT0104_MA_Y',
		'TIC0103_CA_YX',
		'TI0115_MA_Y',
		'TT0652_MA_Y',
		'TIC0103_CA_Y',
		'PIC0104_CA_YX',
		'TIC0101_CA_Y',
		'TIC0105_CA_Y',
		'TT0102_MA_Y',
		'TIC0101_CA_YX',
		'TT0651_MA_Y',
]

# List of column names used as targets
targetColumns = [
    'TT0653_MA_Y'
]

# List of training periods on form ['start', 'end']
traintime = [
        ["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
    ]

# Testing period
testtime = [
    "2018-01-01 00:00:00",
    "2019-05-01 00:00:00"
]

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

# 3. Define models | NB: only RNN (LSTM/GRU) models
lstmd_1x_128 = mlApi.LSTM('lstmr 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500)
gru_1x_128 = mlApi.GRU('gru 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2, training=True, epochs=500)

modelList = [
	lstmd_1x_128,
	gru_1x_128,
]

# 4. Initiate and train models
retrain=False
mlApi.initModels(modelList)
mlApi.trainModels(retrain)

# 5. Predict
predictions, means, stds = mlApi.predictWithModelsUsingDropout()

# 6. Plot predictions for each model
for i in range(len(modelList)):
	mean = means[i]
	std = stds[i]

	import numpy as np

	upper = np.add(mean, std)
	lower = np.subtract(mean, std)

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(1, 1, figsize=(10,3), dpi=100)
	ax.set_xlabel('Date')
	ax.set_ylabel(mlApi.columnDescriptions[targetColumns[0]])
	ax.set_title(modelList[i].name + "\nPredictions and targets")
	ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, y_test[mlApi.maxEnrolWindow:], color="red", alpha=0.5, label="targets")
	ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, upper, color="grey", alpha=0.7, label="+/- 1 std bounds")
	ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, lower, color="grey", alpha=0.7)
	ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, mean, color="blue", alpha=1.0, label="prediction")
	ax.grid(1, axis='y')
	ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10})
    
	plt.show()