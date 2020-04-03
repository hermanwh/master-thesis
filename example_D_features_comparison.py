import src.core as mlApi
from src.utils.plots import (plotModelPredictions, plotModelScores, getPlotColors)
from src.utils.prints import (printModelScores)
import matplotlib.pyplot as plt

colors = getPlotColors()

columnsLists = []
deviationsLists= []
names = []
trainmetrics = []
testmetrics = []

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

targetColumns = [
	'50TT002',
    '20PDT001',
]

models = ['A', 'X', 'B', 'C']

irrelevantColumnsList = [
	# Model A:
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in 
	[
		'20PT001',
		'50PDT001',
		'50FT001',
		'50TV001',
		'50PT001',
	],
	# Model X:
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C flow
	[
		'20PT001',
		'50PDT001',
		'50TV001',
		'50PT001',
	],
	# Model B:
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve
	[
		'20PT001',
		'50PDT001',
		'50FT001',
	],
	# Model C:
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve, C flow
	[
		'20PT001',
		'50PDT001',
	],
]

for i, irrelevantColumns in enumerate(irrelevantColumnsList):
	mlApi.reset()
	df = mlApi.initDataframe(filename, columns, irrelevantColumns)
	df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
	X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)
	linear_model = mlApi.Linear_Regularized("linear model " + models[i])
	mlp_model = mlApi.MLP("mlp " + models[i], layers=[64, 64], dropout=0.2, epochs=500, verbose=0)
	lstm_model = mlApi.LSTM("lstm " + models[i], layers=[64, 64], dropout=0.2, recurrentDropout=0.2, epochs=250, enrolWindow=3)

	modelList = [
		linear_model,
		mlp_model,
		lstm_model,
	]

	mlApi.initModels(modelList)
	retrain=False
	mlApi.trainModels(retrain)

	modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(plot=True)

	if i < 1:
		columnsLists = columnsList
		deviationsLists = deviationsList
		all_names = modelNames
		all_train_metrics = metrics_train
		all_test_metrics = metrics_test
	else:
		for j_target in range(len(columnsList)):
			for k_model in range(1, len(columnsList[j_target])):
				columnsLists[j_target].append(columnsList[j_target][k_model])
			for k_model in range(0, len(deviationsList[j_target])):
				deviationsLists[j_target].append(deviationsList[j_target][k_model])
		all_names = [*all_names, *modelNames]
		all_train_metrics = [*all_train_metrics, *metrics_train]
		all_test_metrics = [*all_test_metrics, *metrics_test]


	names.append(modelNames)
	trainmetrics.append(metrics_train)
	testmetrics.append(metrics_test)

indexColumn = mlApi._indexColumn
columnDescriptions = mlApi._columnDescriptions
columnUnits = mlApi._columnUnits
traintime = mlApi._traintime

for i in range(len(deviationsLists)):
	for j in range(len(deviationsLists[i])):
		deviationsLists[i][j][3] = colors[j]

for i in range(len(columnsLists)):
	for j in range(len(columnsLists[i])):
		columnsLists[i][j][3] = colors[j]

printModelScores(
    all_names,
    all_train_metrics,
    all_test_metrics,
)
plotModelPredictions(
    plt,
    deviationsLists,
    columnsLists,
    indexColumn,
    columnDescriptions,
    columnUnits,
    traintime,
    interpol=False,
)
plotModelScores(
    plt,
    all_names,
    all_train_metrics,
    all_test_metrics,
)