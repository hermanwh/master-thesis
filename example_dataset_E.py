import statApi
from api import Api
mlApi = Api()

# define dataset specifics
filename = "../master-thesis-db/datasets/E/data.csv"

columns = [
	['TT0102': 'Varm side temperatur inn', 'degrees'],
	['TT0107': 'Varm side temperatur ut', 'degrees'],
	['FT0005': 'Varm side gasseksport', 'M^3/h (?)'],
	['FT0161': 'Varm side veske ut av scrubber', 'degrees'],
	['PT0106': 'Varm side trykk innløpsseparator', 'barG'],
	['PDT0105': 'Varm side trykkfall', 'bar'],
	['TT0312': 'Kald side temperatur inn', 'degrees'],
	['TT0601': 'Kald side temperatur ut', 'degrees'],
	['FT0605': 'Kald side kjølemedium', 'M^3/h (?)'],
	['PDT0604': 'Kald side trykkfall', 'bar'],
	['TIC0108': 'Kald side ventilåpning', '%'],
	['HV0175': 'Pumpe bypass', 'unknown'],
	['PT0160': 'Pumpe trykk ut', 'barG'],
]

irrelevantColumns = [
		'HV0175',
		'PT0160',
]

targetColumns = [
    'TT0601',
]

traintime = [
        ["2016-08-01 00:00:00", "2016-11-01 00:00:00"],
]
	
testtime = [
        "2016-01-01 00:00:00",
		"2019-01-01 00:00:00",
	]

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlpd_1x_128 = mlApi.MLP('mlpd 1x 128', layers=[128], dropout=0.2)
lstmd_1x_128 = mlApi.LSTM('lstmr 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2)

linear = mlApi.Linear('linear')
linear_r = mlApi.Linear_Regularized('linear r')

modelList = [
	mlpd_1x_128,
	lstmd_1x_128,
	#linear,
	linear_r,
]

mlApi.initModels(modelList)
retrain=True
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)

