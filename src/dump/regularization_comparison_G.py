import src.core as mlApi

# 1. Define dataset specifics

# File path to dataset .csv file
filename = "../master-thesis-db/datasets/G/data_10min.csv"

# List of columns on form ['name', 'desc', 'unit']
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

# List of column names to ignore completely
irrelevantColumns = [
	'PI0001',
	'FI0027',
	'TIC0022U',
	'PDT0024',
]

# List of column names used a targets
targetColumns = [
	'TT0026',
    'PDI0064',
]

# List of training periods on form ['start', 'end']
traintime = [
	["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
]

# Testing period, recommended: entire dataset
testtime = [
	"2017-01-01 00:00:00",
	"2020-03-01 00:00:00",
]

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlpr1 = mlApi.MLP('MLPr 1x 64 1.0', layers=[64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr2 = mlApi.MLP('MLPr 1x 64 0.5', layers=[64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr3 = mlApi.MLP('MLPr 1x 64 0.1', layers=[64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr4 = mlApi.MLP('MLPr 1x 64 0.05', layers=[64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr5 = mlApi.MLP('MLPr 1x 64 0.01', layers=[64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr6 = mlApi.MLP('MLPr 1x 64 0.005', layers=[64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr7 = mlApi.MLP('MLPr 1x 64 0.001', layers=[64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd = mlApi.MLP('MLPd 1x 64 0.2', layers=[64], dropout=0.2, epochs=5000)

mlpr11 = mlApi.MLP('MLPr 2x 64 1.0', layers=[64, 64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr22 = mlApi.MLP('MLPr 2x 64 0.5', layers=[64, 64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr33 = mlApi.MLP('MLPr 2x 64 0.1', layers=[64, 64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr44 = mlApi.MLP('MLPr 2x 64 0.05', layers=[64, 64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr55 = mlApi.MLP('MLPr 2x 64 0.01', layers=[64, 64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr66 = mlApi.MLP('MLPr 2x 64 0.005', layers=[64, 64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr77 = mlApi.MLP('MLPr 2x 64 0.001', layers=[64, 64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpdd = mlApi.MLP('MLPd 2x 64 0.2', layers=[64, 64], dropout=0.2, epochs=5000)

linear_r = mlApi.Linear_Regularized('linear')

modelList = [
    mlpr1,
    mlpr2,
    mlpr3,
    mlpr4,
    mlpr5,
    mlpr6,
    mlpr7,
	mlpd,
	linear_r,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)

modelList = [
    mlpr5,
    mlpr6,
	mlpd,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)

# --------------

modelList = [
    mlpr11,
    mlpr22,
    mlpr33,
    mlpr44,
    mlpr55,
    mlpr66,
    mlpr77,
	mlpdd,
	linear_r,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)

modelList = [
    mlpr55,
    mlpr66,
	mlpdd,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)