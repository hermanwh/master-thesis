import statApi
from api import Api
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

irrelevantColumns = [
    '50FT001',
    '50PDT001',
    '20PDT001',
]

targetColumns = [
    '50TT002',
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

mlpr1 = mlApi.MLP('MLPr 1x 64 1.0', layers=[64], l1_rate=1.0, l2_rate=1.0, epochs=2500)
mlpr2 = mlApi.MLP('MLPr 1x 64 0.5', layers=[64], l1_rate=0.5, l2_rate=0.5, epochs=2500)
mlpr3 = mlApi.MLP('MLPr 1x 64 0.1', layers=[64], l1_rate=0.1, l2_rate=0.1, epochs=2500)
mlpr4 = mlApi.MLP('MLPr 1x 64 0.05', layers=[64], l1_rate=0.05, l2_rate=0.05, epochs=2500)
mlpr5 = mlApi.MLP('MLPr 1x 64 0.01', layers=[64], l1_rate=0.01, l2_rate=0.01, epochs=2500)
mlpr6 = mlApi.MLP('MLPr 1x 64 0.005', layers=[64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr7 = mlApi.MLP('MLPr 1x 64 0.001', layers=[64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd = mlApi.MLP('MLPd 1x 64', layers=[64], dropout=0.2, epochs=2500)

mlpr11 = mlApi.MLP('MLPr 2x 64 1.0', layers=[64, 64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr22 = mlApi.MLP('MLPr 2x 64 0.5', layers=[64, 64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr33 = mlApi.MLP('MLPr 2x 64 0.1', layers=[64, 64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr44 = mlApi.MLP('MLPr 2x 64 0.05', layers=[64, 64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr55 = mlApi.MLP('MLPr 2x 64 0.01', layers=[64, 64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr66 = mlApi.MLP('MLPr 2x 64 0.005', layers=[64, 64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr77 = mlApi.MLP('MLPr 2x 64 0.001', layers=[64, 64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpdd = mlApi.MLP('MLPd 2x 64', layers=[64, 64], dropout=0.2, epochs=5000)

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
	linear_r,
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)


# -----------

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