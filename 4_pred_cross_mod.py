# Cross model predictions

# %load cross_model_predictions.py
import src.core_stateless as mlModule
import src.core_configs as configs

def trainModelsWithConfig(dirr, mod, res, retrain=False):
    filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(dirr, mod, res)

    relevantColumns, columnDescriptions, columnUnits, columnNames, df = mlModule.initDataframe(filename, columns, irrelevantColumns)
    df = df[columnOrder]
    df_train, df_test = mlModule.getTestTrainSplit(df, traintime, testtime)
    X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(df_train, df_test, targetColumns)

    mlp = mlModule.MLP('MLP 1x128 d0.2 '+dirr+' mod'+mod, X_train, y_train, layers=[128], dropout=0.2)
    lstm = mlModule.LSTM('LSTM 1x128 d0.2 '+dirr+' mod'+mod, X_train, y_train, layers=[128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
    linear = mlModule.Linear_Regularized('Linear rCV ' + dirr + ' mod' + mod, X_train, y_train)

    modelList = [
        mlp,
        lstm,
        linear,
    ]

    maxEnrolWindow, indexColumn = mlModule.initModels(modelList, df_test)
    mlModule.trainModels(modelList, filename, targetColumns, retrain)

    return modelList

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def predictWithConfig(modelList, dirr, mod, res):
	filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(dirr, mod, res)

	relevantColumns, columnDescriptions, columnUnits, columnNames, df = mlModule.initDataframe(filename, columns, irrelevantColumns)
	df = df[columnOrder]
	df_train, df_test = mlModule.getTestTrainSplit(df, traintime, testtime)
	X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(df_train, df_test, targetColumns)
	maxEnrolWindow, indexColumn = mlModule.initModels(modelList, df_test)

	inputScaler = StandardScaler()
	inputScaler.fit(X_train)
	outputScaler = StandardScaler()
	outputScaler.fit(y_train)

	for model in modelList:
		model.inputScaler = inputScaler
		model.outputScaler = outputScaler

	modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
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

def makePredictionsForModAndDir(mod, dirr, dirrs, res, retrain=False):
    allModels = []
    for dirrr in dirrs:
        print("")
        print("Training/loading models for dataset " + dirrr)
        print("Model " + mod)
        print("Resolution " + res)
        print("")
        modelList = trainModelsWithConfig(dirrr, mod, res, retrain)
        allModels.append(modelList)

    for i in range(len(allModels)):
        modelsOfTypei = list(map(lambda x : x[i], allModels))
        print("")
        print("Predictions and deviations for dataset " + dirr)
        print("Model " + mod)
        print("Resolution " + res)
        print("")
        predictWithConfig(modelsOfTypei, dirr, mod, res)
    
makePredictionsForModAndDir('A', 'D', dirrs, res, retrain=False)
makePredictionsForModAndDir('A', 'F', dirrs, res, retrain=False)
makePredictionsForModAndDir('A', 'G', dirrs, res, retrain=False)
makePredictionsForModAndDir('B', 'D', dirrs, res, retrain=False)
makePredictionsForModAndDir('B', 'F', dirrs, res, retrain=False)
makePredictionsForModAndDir('B', 'G', dirrs, res, retrain=False)
