import src.core as mlApi
from src.utils.plots import (plotModelPredictions, plotModelScores, getPlotColors)
from src.utils.prints import (printModelScores)
import matplotlib.pyplot as plt
import src.core_configs as configs

colors = getPlotColors()
models = ['A', 'B', 'C', 'D', 'E']

def featureComparison(
	irrelevantColumnsList,
	filename,
	columns,
	traintime,
	testtime,
	targetColumns,
	enrolWindow,
	):
	global colors, models

	columnsLists = []
	deviationsLists= []
	names = []
	trainmetrics = []
	testmetrics = []

	for i, irrelevantColumns in enumerate(irrelevantColumnsList):
		mlApi.reset()

		df = mlApi.initDataframe(filename, columns, irrelevantColumns)
		df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
		X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)
		
		linear_model = mlApi.Linear_Regularized("Linear " + models[i])
		mlp_model = mlApi.MLP("MLPd " + models[i], layers=[64, 64], dropout=0.2, epochs=2000)
		lstm_model = mlApi.LSTM("LSTMd " + models[i], layers=[64, 64], dropout=0.2, recurrentDropout=0.2, epochs=500, enrolWindow=enrolWindow)

		modelList = [
			linear_model,
			mlp_model,
			lstm_model,
		]

		mlApi.initModels(modelList)
		retrain=False
		mlApi.trainModels(retrain)

		modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(plot=False, score=False)

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
		columnsList[i][0][3] = 'red'
		for j in range(1, len(columnsLists[i])):
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

filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('D', None, '30min')
targetColumns = [
	'50TT002',
	'20PDT001',
]
irrelevantColumnsList = [
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in 
	[
		'20PT001',
		'50PDT001',
		'50FT001',
		'50TV001',
		'50PT001',
	],
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C valve
	[
		'20PT001',
		'50PDT001',
		'50FT001',
		'50PT001',
	],
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C flow
	[
		'20PT001',
		'50PDT001',
		'50TV001',
		'50PT001',
	],
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve
	[
		'20PT001',
		'50PDT001',
		'50FT001',
	],
	#  Target: C T out, P dP
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve, C flow
	[
		'20PT001',
		'50PDT001',
	],
]

featureComparison(irrelevantColumnsList, filename, columns, traintime, testtime, targetColumns, 2)

# -------------------

filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('F', None, '30min')
targetColumns = [
	'TT0653_MA_Y'
]
irrelevantColumnsList = [
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in 
	[
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
		'TT0102_MA_Y',
		'TIC0101_CA_YX',
		'TT0651_MA_Y',
		'TIC0105_CA_Y',
	],
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C valve
	[
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
		'TT0102_MA_Y',
		'TIC0101_CA_YX',
		'TT0651_MA_Y',
	],
]

featureComparison(irrelevantColumnsList, filename, columns, traintime, testtime, targetColumns, 16)

# -------------------------

filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('G', None, '30min')
targetColumns = [
    'TT0026',
	'PDI0064',
]
irrelevantColumnsList = [
	#  Target: C T out, P dP
	#  Features: P T in, P T out, P flow, C T in 
	[
		'PDT0024',
		'FI0027',
		'TIC0022U',
		'PI0001',
	],
	#  Target: C T out, P dP
	#  Features: P T in, P T out, P flow, C T in, C valve
	[
		'PDT0024',
		'FI0027',
		'PI0001',
	],
	#  Target: C T out, P dP
	#  Features: P T in, P T out, P flow, C T in, C flow
	[
		'PDT0024',
		'TIC0022U',
		'PI0001',
	],
	#  Target: C T out, P dP
	#  Features: P T in, P T out, P flow, C T in, C P in, C valve
	[
		'PDT0024',
		'FI0027',
	],
	#  Target: C T out, P dP
	#  Features: P T in, P T out, P flow, C T in, C P in, C valve, C flow
	[
		'PDT0024',
	],
]

featureComparison(irrelevantColumnsList, filename, columns, traintime, testtime, targetColumns, 16)