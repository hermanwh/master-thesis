import src.core as mlApi

# 1. Define dataset specifics

# File path to dataset .csv file
filename = "../master-thesis-db/datasets/B/data_0min.csv"

# List of columns on form ['name', 'desc', 'unit']
columns = [
	['TT181.PV', 'Gas side inlet temperature', 'MSm^3/d'],
	['TIC215.PV', 'Gas side outlet temperature','g/mole'],
	['FI165B.PV', 'Gas side flow', 'degrees'],
	['PT180.PV', 'Gas side compressor pressure', 'degrees'],
	['TT069.PV', 'Cooling side inlet temperature', 'degrees'],
	['PT074.PV', 'Cooling side pressure', 'degrees'],
	['TIC215.OUT', 'Cooling side vavle opening', 'degrees'],
	['XV167.CMD', 'Anti-surge compressor valve', 'degrees'],
	['XV167.ZSH', 'Anti-surge valve', 'degrees'],
	['ZT167.PV', 'Anti-surge unknown', 'Bar'],
]

# List of column names to ignore completely
irrelevantColumns = [
		'XV167.CMD',
		'XV167.ZSH',
		'ZT167.PV',
]

# List of column names used a targets
targetColumns = [
    'TIC215.PV',
]

# List of training periods on form ['start', 'end']
traintime = [
    ["2016-07-01 00:00:00", "2016-10-06 00:00:00"],
]

# Testing period, recommended: entire dataset
testtime = [
	"2016-01-01 00:00:00",
	"2020-03-01 00:00:00",
]

# 2. Initiate and divide data

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

# 3. Define models

mlpd_2x_64 = mlApi.MLP('mlpd 1x 128', layers=[64, 64], dropout=0.2)
lstmd_2x_64 = mlApi.LSTM('lstmr 1x 128', layers=[64, 64], dropout=0.2, recurrentDropout=0.2)
linear_r = mlApi.Linear_Regularized('linear r')
ensemble = mlApi.Ensemble('lstm + mlp ensemble', [lstmd_1x_128, mlpd_1x_128])

modelList = [
	mlpd_2x_64,
	lstmd_2x_64,
    ensemble,
	linear_r,
]

# 4. Initiate and train models

# Define whether to retrain models or not
retrain=False

mlApi.initModels(modelList)
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)

