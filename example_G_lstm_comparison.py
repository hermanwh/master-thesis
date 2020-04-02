import src.core as mlApi

# define dataset specifics
filename = "../master-thesis-db/datasets/G/data_10min.csv"

columns = [
	['PDI0064', 'Process side dP', 'bar'],
	['TI0066', 'Process side Temperature out','degrees'],
	['TZI0012', 'Process side Temperature in', 'degrees'],
	['FI0010', 'Process side flow rate', 'MSm^3/d(?)'],
	['TT0025', 'Cooling side Temperature in', 'degrees'],
	['TT0026', 'Cooling side Temperature out', 'degrees'],
	['PI0001', 'Cooling side Pressure in', 'barG'],
	['FI0027', 'Cooling side flow rate', 'MSm^3/d(?)'],
	['TIC0022U', 'Cooling side valve opening', '%'],
	['PDT0024', 'Cooling side dP', 'bar'],
]

irrelevantColumns = [
	'PI0001',
	'FI0027',
	'TIC0022U',
	'PDT0024',
]

targetColumns = [
	'TT0026',
    'PDI0064',
]

traintime = [
	["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
]

testtime = [
	"2017-01-01 00:00:00",
	"2020-03-01 00:00:00",
]

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

lstm_1x_128 = mlApi.LSTM('lstm 1x 128', layers=[128])

lstmd_1x_16 = mlApi.LSTM('lstmd 1x 16', layers=[16], dropout=0.3)
lstmd_1x_32 = mlApi.LSTM('lstmd 1x 32', layers=[32], dropout=0.3)
lstmd_1x_64 = mlApi.LSTM('lstmd 1x 64', layers=[64], dropout=0.3)
lstmd_1x_128 = mlApi.LSTM('lstmd 1x 128', layers=[128], dropout=0.3)

lstmd_2x_16 = mlApi.LSTM('lstmd 2x 16', layers=[16, 16], dropout=0.3)
lstmd_2x_32 = mlApi.LSTM('lstmd 2x 32', layers=[32, 32], dropout=0.3)
lstmd_2x_64 = mlApi.LSTM('lstmd 2x 64', layers=[64, 64], dropout=0.3)
lstmd_2x_128 = mlApi.LSTM('lstmd 2x 128', layers=[128, 128], dropout=0.3)

linear_cv = mlApi.Linear_Regularized('linear r')

mlp_d = mlApi.MLP('mlp for ensemble 2x 64', layers=[64, 64], dropout=0.2)

ensemble = mlApi.Ensemble('lstmd + linear', [lstmd_2x_64, linear_cv])
ensemble2 = mlApi.Ensemble('lstmd2 + linear', [lstmd_1x_128, linear_cv])
ensemble3 = mlApi.Ensemble('lstm + mlp', [lstmd_2x_64, mlp_d])

modelList = [
    #lstmd_1x_16,
    #lstmd_1x_32,
    #lstmd_1x_64,
    #lstmd_1x_128,
    #lstmd_2x_16,
    #lstmd_2x_32,
    #lstmd_2x_64,
    lstmd_2x_128,
    #ensemble,
    #ensemble2,
	#ensemble3,
    linear_cv,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(plot=True)