import statApi
from api import Api
mlApi = Api()

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
		#'TIC0101_CA_Y',
		#'TIC0105_CA_Y',
]

targetColumns = [
    'TT0651_MA_Y',
    'TT0653_MA_Y'
]

traintime = [
        ["2017-08-05 00:00:00", "2018-08-01 00:00:00"],
    ]

testtime = [
    "2017-08-05 00:00:00",
    "2020-02-01 00:00:00"
]

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

covmat = statApi.correlationMatrix(df_train)
statApi.printCorrelationMatrix(covmat, df_train, mlApi.columnDescriptions)

pca = statApi.pca(df_train, -1, mlApi.relevantColumns, mlApi.columnDescriptions)
statApi.printExplainedVarianceRatio(pca)
"""
statApi.pcaPlot(df, [traintime, testtime, []])

statApi.pairplot(df)

statApi.scatterplot(df)

statApi.correlationPlot(df_train)

statApi.valueDistribution(df, traintime, testtime)

"""

"""

mlp_10 = mlApi.MLP('MLP 10', layers=[10], verbose=0, batchSize=128)
mlp_20 = mlApi.MLP('MLP 20', layers=[20], verbose=0, batchSize=128)
mlp_128 = mlApi.MLP('MLP 128', layers=[128], verbose=0, batchSize=128)
mlp_10_reg = mlApi.MLP_Regularized('MLPr 10', layers=[10], verbose=0, batchSize=128)
mlp_20_reg = mlApi.MLP_Regularized('MLPr 20', layers=[20], verbose=0, batchSize=128)
mlp_128_reg = mlApi.MLP_Regularized('MLPr 128', layers=[128], verbose=0, batchSize=128)
linear = mlApi.Linear('Linear')
linear_reg = mlApi.Linear_Regularized('Linear r')
ensemble = mlApi.Ensemble('Ensemble', [mlp_128_reg, linear_reg])

lstm_128 = mlApi.LSTM('lstm 128', verbose=1, dropout=0.2, alpha=None)
lstm_128_leaky = mlApi.LSTM('lstm 128 leaky', verbose=1, dropout=0.2, alpha=None)
lstm_128_recurrent = mlApi.LSTM_Recurrent('lstm 128 recurrent', verbose=1, dropout=0.2, recurrentDropout=0.0, alpha=None)
lstm_2x_128 = mlApi.LSTM('lstm 2x128', verbose=1, units=[128, 128])
lstm_2x_128_recurrent = mlApi.LSTM_Recurrent('lstm 2x128 recurrent', verbose=1, units=[128, 128])

gru_128_leaky = mlApi.GRU('gru 128 leaky', verbose=1, dropout=0.2, alpha=None)
gru_128_recurrent = mlApi.LSTM_Recurrent('gru 128 recurrent', verbose=1, dropout=0.2, recurrentDropout=0.0, alpha=None)

antoenc_1 = mlApi.Autoencoder_Dropout('autoenc dropout')
autoenc_2 = mlApi.Autoencoder_Regularized('autoenc regularized')

mlp_128_dropout = mlApi.MLP_Dropout('mlp 128 dropout', layers=[128], dropout=0.2, )

mlp_3x_128 = mlApi.MLP_Dropout('mlp 3x', layers=[128, 128, 128], dropout=0.2)
lstm_3x_128 = mlApi.LSTM_Recurrent('lstm 3x', units=[128, 128, 128], dropout=0.2, recurrentDropout=0.2, training=False)

"""

mlp_1x_128 = mlApi.MLP('mlp 1x 128', layers=[128])
mlpd_1x_128 = mlApi.MLP('mlpd 1x 128', layers=[128], dropout=0.2)
mlpr_1x_128 = mlApi.MLP('mlpr 1x 128', layers=[128], l1_rate=0.01, l2_rate=0.01)

lstm_1x_128 = mlApi.LSTM('lstm 1x 128', layers=[128])
lstmd_1x_128 = mlApi.LSTM('lstmr 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2)

modelList = [
	mlp_1x_128,
	mlpd_1x_128,
	mlpr_1x_128,
	lstm_1x_128,
	lstmd_1x_128,
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
modelNames, metrics_train, metrics_test = mlApi.predictWithModels(plot=True, interpol=True)

