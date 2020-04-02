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
filename = "../master-thesis-db/datasets/G/data_30min.csv"

columns = [
	['PDI0064', 'Process side dP', 'bar'],
	['TI0066', 'Process side Temperature out','degrees'],
	['TZI0012', 'Process side Temperature in', 'degrees'],
	['FI0010', 'Process side flow rate', 'MSm^3/d(?)'],
	['TT0025', 'Cooling side Temperature in', 'degrees'],
	['TT0026', 'Cooling side Tmperature out', 'degrees'],
	['PI0001', 'Cooling side Pressure in', 'barG'],
	['FI0027', 'Cooling side flow rate', 'MSm^3/d(?)'],
	['TIC0022U', 'Cooling side valve opening', '%'],
	['PDT0024', 'Cooling side dP', 'bar'],
]

traintime = [
	["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
]

testtime = [
	"2017-01-01 00:00:00",
	"2020-03-01 00:00:00",
]

targetColumns = [
    'TT0026'
]

models = ['A', 'B', 'C']

irrelevantColumnsList = [
	# Model A:
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in 
	[
		'PDI0064',
		'PDT0024',
		'FI0027',
		'TIC0022U',
		'PI0001',
	],
	# Model B:
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve
	[
		'PDI0064',
		'PDT0024',
		'FI0027',
	],
	# Model C:
	#  Target: C T out
	#  Featers: P T in, P T out, P flow, C T in, C P in, C valve, C flow
	[
		'PDI0064',
		'PDT0024',
	],
]

for i, irrelevantColumns in enumerate(irrelevantColumnsList):
    mlApi.reset()
    df = mlApi.initDataframe(filename, columns, irrelevantColumns)
    df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
    X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)
    linear_model = mlApi.Linear_Regularized("linear model " + models[i])
    mlp_model = mlApi.MLP("mlp " + models[i], layers=[128], dropout=0.2, epochs=1000, verbose=0)
    lstm_model = mlApi.LSTM("lstm " + models[i], layers=[64, 64], dropout=0.2, recurrentDropout=0.2, epochs=250, enrolWindow=18)
    
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