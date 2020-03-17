from api import Api
mlApi = Api()

# define dataset specifics
filename = "../HX-condition-monitoring/datasets/D/data.csv"

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
    '50TV001',
]

targetColumns = [
    '50TT002',
]

traintime = [
        ["2020-01-01 00:00:00", "2020-02-01 00:00:00"],
    ]

testtime = [
    "2020-01-01 00:00:00",
    "2020-07-01 00:00:00"
]

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)


mlp_10 = mlApi.MLP('MLP 10', layers=[10])
mlp_20 = mlApi.MLP('MLP 20', layers=[20])
mlp_128 = mlApi.MLP('MLP 128', layers=[128])
mlp_10_reg = mlApi.MLP_Regularized('MLPr 10', layers=[10])
mlp_20_reg = mlApi.MLP_Regularized('MLPr 20', layers=[20])
mlp_128_reg = mlApi.MLP_Regularized('MLPr 128', layers=[128])
linear = mlApi.Linear('Linear')
linear_reg = mlApi.Linear_Regularized('Linear r')
ensemble = mlApi.Ensemble('Ensemble', [mlp_128_reg, linear_reg])


lstm_128 = mlApi.LSTM('lstm  128')
lstm_128_recurrent = mlApi.LSTM_Recurrent('lstm 128 recurrent')
lstm_2x_128 = mlApi.LSTM('lstm 2x128', units=[128, 128])
lstm_2x_128_recurrent = mlApi.LSTM_Recurrent('lstm 2x128 recurrent', units=[128, 128])

modelList = [
    mlp_128,
    mlp_128_reg,
    linear_reg,
    ensemble
]

mlApi.initModels(modelList)
retrain=False
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test = mlApi.predictWithModels(plot=True)