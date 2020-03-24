import statApi
from api import Api
mlApi = Api()

# define dataset specifics
filename = "../master-thesis-db/datasets/B/data_0min.csv"

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

irrelevantColumns = [
		'XV167.CMD',
		'XV167.ZSH',
]

targetColumns = [
    'TIC215.PV',
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

covmat = statApi.correlationMatrix(df_train)
statApi.printCorrelationMatrix(covmat, df_train, mlApi.columnDescriptions)

pca = statApi.pca(df_train, -1, mlApi.relevantColumns, mlApi.columnDescriptions)
statApi.printExplainedVarianceRatio(pca)

#mlp_1x_128 = mlApi.MLP('mlp 1x 128', layers=[128])
mlpd_1x_128 = mlApi.MLP('mlpd 1x 128', layers=[128], dropout=0.2)
#mlpr_1x_128 = mlApi.MLP('mlpr 1x 128', layers=[128], l1_rate=0.01, l2_rate=0.01)

#lstm_1x_128 = mlApi.LSTM('lstm 1x 128', layers=[128])
lstmd_1x_128 = mlApi.LSTM('lstmr 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2)

linear = mlApi.Linear('linear')
linear_r = mlApi.Linear_Regularized('linear r')

modelList = [
	lstmd_1x_128,
	#mlp_1x_128,
	mlpd_1x_128,
	#mlpr_1x_128,
	#lstmd_1x_128,
	#linear,
	linear_r,
]

mlApi.initModels(modelList)
retrain=True
mlApi.trainModels(retrain)

import src.utils.modelFuncs as mf

for model in modelList:
	print(model.name)
	mf.printModelSummary(model)
	print("")
	print("")

#mlApi.predictWithAutoencoderModels()
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(plot=True, interpol=True)

