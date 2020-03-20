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
    '20PDT001',
    '50PDT001',
]

targetColumns = [
    '50TT002',
]

traintime = [
        ["2020-01-01 00:00:00", "2020-03-01 00:00:00"],
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
mlp_2x_128 = mlApi.MLP('MLP 2x128', layers=[128, 128])
mlp_10x_128 = mlApi.MLP('MLP 10x128', layers=[128, 128, 128, 128, 128, 128, 128, 128, 128, 128])
mlp_10_reg = mlApi.MLP_Regularized('MLPr 10', layers=[10])
mlp_20_reg = mlApi.MLP_Regularized('MLPr 20', layers=[20])
mlp_128_reg = mlApi.MLP_Regularized('MLPr 128', layers=[128])
linear = mlApi.Linear('Linear')
linear_reg = mlApi.Linear_Regularized('Linear r')
ensemble = mlApi.Ensemble('Ensemble', [mlp_128_reg, linear_reg])

lstm_128 = mlApi.LSTM('lstm  128', dropout=0.5)
lstm_128_recurrent = mlApi.LSTM_Recurrent('lstm 128 recurrent', dropout=0.5, recurrentDropout=0.5)
lstm_2x_128 = mlApi.LSTM('lstm 2x128', units=[128, 128])
lstm_2x_128_recurrent = mlApi.LSTM_Recurrent('lstm 2x128 recurrent', units=[128, 128])

mlp_128_reg_1 = mlApi.MLP_Regularized('MLPr 128 reg 1', layers=[128], l1_rate=0.5, l2_rate=0.5)
mlp_128_reg_2 = mlApi.MLP_Regularized('MLPr 128 reg 2', layers=[128], l1_rate=0.1, l2_rate=0.1)
mlp_128_reg_3 = mlApi.MLP_Regularized('MLPr 128 reg 3', layers=[128], l1_rate=0.05, l2_rate=0.05)
mlp_128_reg_4 = mlApi.MLP_Regularized('MLPr 128 reg 4', layers=[128], l1_rate=0.01, l2_rate=0.01)
mlp_128_reg_5 = mlApi.MLP_Regularized('MLPr 128 reg 5', layers=[128], l1_rate=0.005, l2_rate=0.005)
mlp_128_reg_6 = mlApi.MLP_Regularized('MLPr 128 reg 6', layers=[128], l1_rate=0.001, l2_rate=0.001)

mlp_2x_128_reg = mlApi.MLP_Regularized('MLPr 2x128 reg 4', layers=[128, 128], l1_rate=0.01, l2_rate=0.01)
mlp_10x_128_reg = mlApi.MLP_Regularized('MLPr 10x128 reg 4', layers=[128, 128, 128, 128, 128, 128, 128, 128, 128, 128], l1_rate=0.01, l2_rate=0.01)

antoenc_1 = mlApi.Autoencoder_Dropout('autoenc dropout')
autoenc_2 = mlApi.Autoencoder_Regularized('autoenc regularized')

modelList = [
    antoenc_1,
    autoenc_2,
    #mlp_10,
    #mlp_20,
    #mlp_128,
    #mlp_2x_128,
    #mlp_128_reg_1,
    #mlp_128_reg_2,
    #mlp_128_reg_3,
    #mlp_128_reg_4,
    #mlp_128_reg_5,
    #mlp_128_reg_6,
    #mlp_10x_128,
    #mlp_2x_128_reg,
    #mlp_10x_128_reg,
    #linear_reg,
]

mlApi.initModels(modelList)
retrain=True
mlApi.trainModels(retrain)
mlApi.predictWithAutoencoderModels()
#modelNames, metrics_train, metrics_test = mlApi.predictWithModels(plot=True)