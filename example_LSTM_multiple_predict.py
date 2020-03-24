from api import Api
mlApi = Api()

# define dataset specifics
filename = "../master-thesis-db/datasets/F/data_60min.csv"

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

irrelevantColumns = [
		'FT0111',
		'PDT0108_MA_Y',
		'PDT0119_MA_Y',
		'PDT0118_MA_Y',
		'TT0104_MA_Y',
		'TIC0103_CA_YX',
		'TT0652_MA_Y',
		'TIC0103_CA_Y',
        'TT0102_MA_Y',
        'TIC0101_CA_YX',
		'TIC0101_CA_Y',
        'TT0651_MA_Y',
		#'TIC0105_CA_Y',
]

targetColumns = [
    'TT0653_MA_Y',
]

traintime = [
        ["2017-08-05 00:00:00", "2018-08-01 00:00:00"],
    ]

testtime = [
    "2017-08-05 00:00:00",
    "2020-02-01 00:00:00"
]


df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

lstm_128 = mlApi.LSTM('lstm  128', dropout=0.5, enrolWindow=24)
lstm_128_recurrent = mlApi.LSTM_Recurrent('lstm 128 recurrent', dropout=0.5, recurrentDropout=0.5, enrolWindow=24)
lstm_2x_128 = mlApi.LSTM('lstm 2x128', units=[128, 128])
lstm_2x_128_recurrent = mlApi.LSTM_Recurrent('lstm 2x128 recurrent', units=[128, 128])

modelList = [
    lstm_128,
    lstm_128_recurrent,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)

import numpy as np
pred = lstm_128.predict(X_test, y_test)
predictions, mean, std = lstm_128_recurrent.predictMultiple(X_test, y_test)
upper = np.add(mean, std)
lower = np.subtract(mean, std)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel('Date')
ax.set_ylabel(targetColumns[0])

ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, upper, color="grey", alpha=0.5)
ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, lower, color="grey", alpha=0.5)
ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, mean, color="red", alpha=0.9, label="prediction")
ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, y_test[mlApi.maxEnrolWindow:], color="blue", alpha=0.9, label="targets")
ax.plot(df_test.iloc[mlApi.maxEnrolWindow:].index, pred, color="black", alpha=0.9, label="lstm 128")
ax.grid(1, axis='y')
ax.legend(loc='upper left')

plt.show()

#modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(plot=True)