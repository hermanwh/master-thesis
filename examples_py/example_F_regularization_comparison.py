import api as mlApi

# define dataset specifics
filename = "../master-thesis-db/datasets/F/data2_30min.csv"

columns = [
	['FYN0111', 'Gasseksport rate', 'MSm^3/d'],
	['FT0111', 'Gasseksport molvekt','g/mole'],
	['TT0102_MA_Y', 'Varm side A temperatur inn', 'degrees'],
	['TIC0101_CA_YX', 'Varm side A temperatur ut', 'degrees'],
	['TT0104_MA_Y', 'Varm side B temperatur inn', 'degrees'],
	['TIC0103_CA_YX', 'Varm side B temperatur ut', 'degrees'],
	['TT0106_MA_Y', 'Varm side C temperatur inn', 'degrees'],
	['TIC0105_CA_YX', 'Varm side C temperatur ut', 'degrees'],
	['TI0115_MA_Y', 'Scrubber temperatur ut', 'degrees'],
	['PDT0108_MA_Y', 'Varm side A trykkfall', 'Bar'],
	['PDT0119_MA_Y', 'Varm side B trykkfall', 'Bar'],
	['PDT0118_MA_Y', 'Varm side C trykkfall', 'Bar'],
	['PIC0104_CA_YX', 'Innløpsseparator trykk', 'Barg'],
	['TIC0425_CA_YX', 'Kald side temperatur inn', 'degrees'],
	['TT0651_MA_Y', 'Kald side A temperatur ut', 'degrees'],
	['TT0652_MA_Y', 'Kald side B temperatur ut', 'degrees'],
	['TT0653_MA_Y', 'Kald side C temperatur ut', 'degrees'],
	['TIC0101_CA_Y', 'Kald side A ventilåpning', '%'],
	['TIC0103_CA_Y', 'Kald side B ventilåpning', '%'],
	['TIC0105_CA_Y', 'Kald side C ventilåpning', '%'],
]

irrelevantColumns = [
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
]

targetColumns = [
	'TT0653_MA_Y',
]

traintime = [
        ["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
    ]

testtime = [
    "2018-01-01 00:00:00",
    "2019-05-01 00:00:00"
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
mlpd = mlApi.MLP('MLPd 1x 64', layers=[64], dropout=0.2, epochs=5000)

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