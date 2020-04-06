import src.core_stateless as statelessApi
import src.core_configs as configs

def trainModelsWithConfig(dirr, mod, res):
	filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(dirr, mod, res)

	relevantColumns, columnDescriptions, columnUnits, columnNames, df = statelessApi.initDataframe(filename, columns, irrelevantColumns)
	df = df[columnOrder]
	df_train, df_test = statelessApi.getTestTrainSplit(df, traintime, testtime)
	X_train, y_train, X_test, y_test = statelessApi.getFeatureTargetSplit(df_train, df_test, targetColumns)

	mlpd_2x_64 = statelessApi.MLP('mlpd 2x64 ' + dirr + ' ' + mod + ' ' + res, X_train, y_train, layers=[64, 64], dropout=0.2)
	lstmd_2x_64 = statelessApi.LSTM('lstmd 2x64 ' + dirr + ' ' + mod + ' ' + res, X_train, y_train, layers=[64, 64], dropout=0.2, recurrentDropout=0.2)
	linear_r = statelessApi.Linear_Regularized('linear ' + dirr + ' ' + mod + ' ' + res, X_train, y_train)

	modelList = [
		mlpd_2x_64,
		lstmd_2x_64,
		linear_r,
	]

	retrain=False

	maxEnrolWindow, indexColumn = statelessApi.initModels(modelList, df_test)
	statelessApi.trainModels(modelList, filename, targetColumns, retrain)

	return modelList

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def predictWithConfig(modelList, dirr, mod, res):
	filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(dirr, mod, res)

	relevantColumns, columnDescriptions, columnUnits, columnNames, df = statelessApi.initDataframe(filename, columns, irrelevantColumns)
	df = df[columnOrder]
	df_train, df_test = statelessApi.getTestTrainSplit(df, traintime, testtime)
	X_train, y_train, X_test, y_test = statelessApi.getFeatureTargetSplit(df_train, df_test, targetColumns)
	maxEnrolWindow, indexColumn = statelessApi.initModels(modelList, df_test)

	inputScaler = StandardScaler()
	inputScaler.fit(X_train)
	outputScaler = StandardScaler()
	outputScaler.fit(y_train)

	for model in modelList:
		model.inputScaler = inputScaler
		model.outputScaler = outputScaler

	modelNames, metrics_train, metrics_test, columnsList, deviationsList = statelessApi.predictWithModels(
		modelList,
		X_train,
		y_train,
		X_test,
		y_test,
		targetColumns,
		indexColumn,
		columnDescriptions,
		columnUnits,
		traintime,
		plot=True,
		interpol=False,
	)

dirrs = ['D', 'F', 'G']
mods = ['A', 'B']
res = '30min'

for mod in mods:
	allModels = []
	for dirr in dirrs:
		modelList = trainModelsWithConfig(dirr, mod, res)
		allModels.append(modelList)

	for i in range(len(allModels)):
		modelsOfTypei = list(map(lambda x : x[i], allModels))
		for dirr in dirrs:
			print("")
			print("Predictions for dataset " + dirr)
			print("Model " + mod)
			print("Resolution " + res)
			print("")
			predictWithConfig(modelsOfTypei, dirr, mod, res)