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

mlp_1x_128 = mlApi.MLP('mlp 1x 128', layers=[128])

mlpd_1x_16 = mlApi.MLP('mlpd 1x 16', layers=[16], dropout=0.3)
mlpd_1x_32 = mlApi.MLP('mlpd 1x 32', layers=[32], dropout=0.3)
mlpd_1x_64 = mlApi.MLP('mlpd 1x 64', layers=[64], dropout=0.3)
mlpd_1x_128 = mlApi.MLP('mlpd 1x 128', layers=[128], dropout=0.3)

mlpd_2x_16 = mlApi.MLP('mlpd 2x 16', layers=[16, 16], dropout=0.3)
mlpd_2x_32 = mlApi.MLP('mlpd 2x 32', layers=[32, 32], dropout=0.3)
mlpd_2x_64 = mlApi.MLP('mlpd 2x 64', layers=[64, 64], dropout=0.3)
mlpd_2x_128 = mlApi.MLP('mlpd 2x 128', layers=[128, 128], dropout=0.3)

mlpr_1x_128 = mlApi.MLP('mlpr 1x 128', layers=[128], l1_rate=0.01, l2_rate=0.01)

linear = mlApi.Linear('linear')

lstm_1x_128 = mlApi.LSTM('lstm 1x 128', layers=[128])
lstmd_1x_128 = mlApi.LSTM('lstmr 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2)
"""
pca = statApi.pca(df_train, -1, mlApi.relevantColumns, mlApi.columnDescriptions)
statApi.printExplainedVarianceRatio(pca)

statApi.pcaPlot(df, [traintime, testtime, []])

statApi.pairplot(df)

statApi.scatterplot(df)

covmat_train = statApi.correlationMatrix(df_train)
statApi.printCorrelationMatrix(covmat_train, df_train, mlApi.columnDescriptions)
statApi.correlationPlot(df_train)

covmat_test = statApi.correlationMatrix(df_test)
statApi.printCorrelationMatrix(covmat_test, df_test, mlApi.columnDescriptions)


statApi.correlationPlot(df_test)

statApi.valueDistribution(df, traintime, testtime)
"""
ensemble = mlApi.Ensemble(
    'linear + mlpd128',
    [
        linear,
        mlpd_1x_16,
        mlpd_1x_32,
        mlpd_1x_64,
        mlpd_1x_128,
    ]
)

modelList = [
	#mlp_1x_128,
    #mlpd_1x_16,
    #mlpd_1x_32,
    #mlpd_1x_64,
	mlpd_1x_128,
    #mlpd_2x_16,
    #mlpd_2x_32,
    #mlpd_2x_64,
	#mlpd_2x_128,
	#mlpr_1x_128,
	lstm_1x_128,
	lstmd_1x_128,
    #ensemble,
    linear,
]

mlApi.initModels(modelList)
retrain=True
mlApi.trainModels(retrain)

#mlApi.predictWithAutoencoderModels()
modelNames, metrics_train, metrics_test = mlApi.predictWithModels(plot=True)

print(linear.model.coef_)