from api import Api
from src.utils.plots import (plotModelPredictions, plotModelScores, getPlotColors)
from src.utils.prints import (printModelScores)
import matplotlib.pyplot as plt

colors = getPlotColors()

targetColumns = [
	'50TT002',
]

colList = []
devList = []
colTarget = None
names = []
trainmetrics = []
testmetrics = []

irrelevantColumnsList = [
	[
		'asd'
	],
	[
		'20PT001',
		'50PT001',
	],
	[
		'20PT001',
		'50PT001',
		'20PDT001',
		'50PDT001',
	],
	[
		'20PT001',
		'50PT001',
		'20PDT001',
		'50PDT001',
		'50FT001',
		'50TV001',
	],
]

for i, irrelevantColumns in enumerate(irrelevantColumnsList):

	mlApi = Api()

	# define dataset specifics
	filename = "../master-thesis-db/datasets/D/dataC.csv"

	columns = [
		['20TT001', 'Gas side inlet temperature', 'degrees'],
		['20PT001', 'Gas side inlet pressure', 'barG'],
		['20FT001', 'Gas side flow', 'M^3/s'],
		['20TT002', 'Gas side outlet temperature', 'degrees'],
		['20PDT001', 'Gas side pressure difference', 'bar'],
		['50TT001', 'Cooling side inlet temperature', 'degrees'],
		['50PT001', 'Cooling side inlet pressure', 'barG'],
		['50FT001', 'Cooling side flow', 'M^3/s'],
		['50TT002', 'Cooling side outlet temperature', 'degrees'],
		['50PDT001', 'Cooling side pressure differential', 'bar'],
		['50TV001', 'Cooling side valve opening', '%'],
	]

	traintime = [
			["2020-01-01 00:00:00", "2020-04-01 00:00:00"],
		]

	testtime = [
		"2020-01-01 00:00:00",
		"2020-08-01 00:00:00"
	]

	df = mlApi.initDataframe(filename, columns, irrelevantColumns)
	df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
	X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

	linear_model = mlApi.Linear_Regularized("linear model " + str(i))
	mlp_model = mlApi.MLP("mlp " + str(i), layers=[128], dropout=0.2, epochs=500, verbose=0)

	modelList = [
		mlp_model,
	]

	mlApi.initModels(modelList)
	retrain=True
	mlApi.trainModels(retrain)

	#mlApi.predictWithAutoencoderModels()
	modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(plot=False)
	
	colTarget = columnsList[0][0]
	colList.append(columnsList[0][1])
	devList.append(deviationsList[0][0])
	names.append(modelNames[0])
	trainmetrics.append(metrics_train[0])
	testmetrics.append(metrics_test[0])

	#print(linear_model.model.coef_)

	indexColumn = mlApi.indexColumn
	columnDescriptions = mlApi.columnDescriptions
	columnUnits = mlApi.columnUnits
	traintime = mlApi.traintime

for i in range(len(devList)):
	devList[i][3] = colors[i]

for i in range(len(colList)):
	colList[i][3] = colors[i]

printModelScores(
	names,
	trainmetrics,
	testmetrics
)
plotModelPredictions(
	plt,
	[devList],
	[[colTarget, *colList]],
	indexColumn,
	columnDescriptions,
	columnUnits,
	traintime,
	interpol=False,
)
plotModelScores(
	plt,
	names,
	trainmetrics,
	testmetrics
)