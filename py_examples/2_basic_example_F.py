# basic F

import src.core as mlModule

model = 'A'

# File path to dataset .csv file
filename = "../master-thesis-db/datasets/F/data_30min.csv"

# List of columns on form ['name', 'description', 'unit']
columns = [
    ['FYN0111', 'Process Flow Rate', 'MSm^3/day'],
    ['FT0111', 'Process Flow Molecular Weight','g/mole'],
    ['TT0102_MA_Y', 'Process Inlet Temperature A', 'Degrees'],
    ['TIC0101_CA_YX', 'Process Outlet Temperature A', 'Degrees'],
    ['TT0104_MA_Y', 'Process Inlet Temperature B', 'Degrees'],
    ['TIC0103_CA_YX', 'Process Outlet Temperature B', 'Degrees'],
    ['TT0106_MA_Y', 'Process Inlet Temperature C', 'Degrees'],
    ['TIC0105_CA_YX', 'Process Outlet Temperature C', 'Degrees'],
    ['TI0115_MA_Y', 'Scrubber Outlet Temperature', 'Degrees'],
    ['PDT0108_MA_Y', 'Process A Pressure Difference', 'Bar'],
    ['PDT0119_MA_Y', 'Process B Pressure Difference', 'Bar'],
    ['PDT0118_MA_Y', 'Process C Pressure Difference', 'Bar'],
    ['PIC0104_CA_YX', 'Separator Inlet Pressure', 'Bar'],
    ['TIC0425_CA_YX', 'Coolant Inlet Temperature', 'Degrees'],
    ['TT0651_MA_Y', 'Coolant Outlet Temperature A', 'Degrees'],
    ['TT0652_MA_Y', 'Coolant Outlet Temperature B', 'Degrees'],
    ['TT0653_MA_Y', 'Coolant Outlet Temperature C', 'Degrees'],
    ['TIC0101_CA_Y', 'Coolant Valve Opening A', '%'],
    ['TIC0103_CA_Y', 'Coolant Valve Opening B', '%'],
    ['TIC0105_CA_Y', 'Coolant Valve Opening C', '%'],
]

# List of column names to ignore completely
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
		'TIC0105_CA_Y',
]

# List of column names used a targets
targetColumns = [
	'TT0653_MA_Y',
]


# List of training periods on form ['start', 'end']
traintime = [
	["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
]

# Testing period, recommended: entire dataset
testtime = [
    "2018-01-01 00:00:00",
    "2019-05-01 00:00:00"
]

df = mlModule.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlModule.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(targetColumns)

mlp_1 = mlModule.MLP('MLP 1x64 d0.2 mod'+model, layers=[64], dropout=0.2)
mlp_2 = mlModule.MLP('MLP 1x128 d0.2 mod'+model, layers=[128], dropout=0.2)
mlp_3 = mlModule.MLP('MLP 2x64 d0.2 mod'+model, layers=[64, 64], dropout=0.2)
mlp_4 = mlModule.MLP('MLP 2x128 d0.2 mod'+model, layers=[128, 128], dropout=0.2)
lstm_1 = mlModule.LSTM('LSTM 1x64 d0.2 mod'+model, layers=[64], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
lstm_2 = mlModule.LSTM('LSTM 1x128 d0.2 mod'+model, layers=[128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
lstm_3 = mlModule.LSTM('LSTM 2x64 d0.2 mod'+model, layers=[64, 64], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
lstm_4 = mlModule.LSTM('LSTM 2x128 d0.2 mod'+model, layers=[128, 128], dropout=0.2, recurrentDropout=0.2, enrolWindow=12)
linear = mlModule.Linear_Regularized('Linear rCV mod'+model)
ensemble1 = mlModule.Ensemble('MLP 1x128 + Linear mod'+model, [mlp_2, linear])
ensemble2 = mlModule.Ensemble('LSTM 1x128 + Linear mod'+model, [lstm_2, linear])

modelList = [
    linear,
    mlp_1,
    mlp_2,
    mlp_3,
    mlp_4,
    lstm_1,
    lstm_2,
    lstm_3,
    lstm_4,
    ensemble1,
    ensemble2,
]

# Define whether to retrain models or not
retrain=False

mlModule.initModels(modelList)
mlModule.trainModels(retrain)

modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
	plot=True,
	interpol=False,
	score=True,
)

for model in modelList:
    mlModule.initModels([model])
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
        plot=True,
        interpol=False,
        score=False,
    )