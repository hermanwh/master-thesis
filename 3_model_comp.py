# Features comparison

# %load features_comparison.py
import src.core as mlModule
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
		mlModule.reset()

		df = mlModule.initDataframe(filename, columns, irrelevantColumns)
		df_train, df_test = mlModule.getTestTrainSplit(traintime, testtime)

		X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(targetColumns)

		mlp_1 = mlModule.MLP('MLP 1x64 d0.2 mod'+models[i], layers=[64], dropout=0.2)
		mlp_2 = mlModule.MLP('MLP 1x128 d0.2 mod'+models[i], layers=[128], dropout=0.2)
		mlp_3 = mlModule.MLP('MLP 2x64 d0.2 mod'+models[i], layers=[64, 64], dropout=0.2)
		mlp_4 = mlModule.MLP('MLP 2x128 d0.2 mod'+models[i], layers=[128, 128], dropout=0.2)
		lstm_1 = mlModule.LSTM('LSTM 1x64 d0.2 mod'+models[i], layers=[64], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
		lstm_2 = mlModule.LSTM('LSTM 1x128 d0.2 mod'+models[i], layers=[128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
		lstm_3 = mlModule.LSTM('LSTM 2x64 d0.2 mod'+models[i], layers=[64, 64], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
		lstm_4 = mlModule.LSTM('LSTM 2x128 d0.2 mod'+models[i], layers=[128, 128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
		linear = mlModule.Linear_Regularized('Linear rCV mod'+models[i])

		modelList = [
			mlp_1,
			mlp_2,
			mlp_3,
			mlp_4,
			lstm_1,
			lstm_2,
			lstm_3,
			lstm_4,
			linear,
		]

		mlModule.initModels(modelList)
		retrain=False
		mlModule.trainModels(retrain)

		modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(plot=True, score=True)

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

	indexColumn = mlModule._indexColumn
	columnDescriptions = mlModule._columnDescriptions
	columnUnits = mlModule._columnUnits
	traintime = mlModule._traintime

	for i in range(len(deviationsLists)):
		for j in range(len(deviationsLists[i])):
			deviationsLists[i][j][3] = colors[j]

	for i in range(len(columnsLists)):
		columnsList[i][0][3] = 'red'
		for j in range(1, len(columnsLists[i])):
			columnsLists[i][j][3] = colors[j-1]

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
]
irrelevantColumnsList = [
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in 
	[
		'20PT001',
		'50PDT001',
		'50FT001',
		'50TV001',
		'50PT001',
	    '20PDT001',
	],
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C valve
	[
		'20PT001',
		'50PDT001',
		'50FT001',
		'50PT001',
	    '20PDT001',
	],
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C flow
	[
		'20PT001',
		'50PDT001',
		'50TV001',
		'50PT001',
	    '20PDT001',
	],
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve
	[
		'20PT001',
		'50PDT001',
		'50FT001',
	    '20PDT001',
	],
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve, C flow
	[
		'20PT001',
		'50PDT001',
	    '20PDT001',
	],
]

featureComparison(irrelevantColumnsList, filename, columns, traintime, testtime, targetColumns, 2)

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

filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('G', None, '30min')
targetColumns = [
    'TT0026',
]
irrelevantColumnsList = [
	#  Target: C T out
	#  Features: P T in, P T out, P flow, C T in 
	[
		'PDT0024',
		'FI0027',
		'TIC0022U',
		'PI0001',
	    'PDI0064',
	],
	#  Target: C T out
	#  Features: P T in, P T out, P flow, C T in, C valve
	[
		'PDT0024',
		'FI0027',
		'PI0001',
	    'PDI0064',
	],
	#  Target: C T out
	#  Features: P T in, P T out, P flow, C T in, C flow
	[
		'PDT0024',
		'TIC0022U',
		'PI0001',
	    'PDI0064',
	],
	#  Target: C T out
	#  Features: P T in, P T out, P flow, C T in, C P in, C valve
	[
		'PDT0024',
		'FI0027',
	    'PDI0064',
	],
	#  Target: C T out
	#  Features: P T in, P T out, P flow, C T in, C P in, C valve, C flow
	[
		'PDT0024',
	    'PDI0064',
	],
]

featureComparison(irrelevantColumnsList, filename, columns, traintime, testtime, targetColumns, 16)