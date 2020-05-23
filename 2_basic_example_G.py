# basic G

import src.core as mlModule

model = 'A'

# File path to dataset .csv file
filename = "../master-thesis-db/datasets/G/data_30min.csv"

# List of columns on form ['name', 'description', 'unit']
columns = [
    ['PDI0064', 'Process Pressure Difference', 'Bar'],
    ['TI0066', 'Process Outlet Temperature','Degrees'],
    ['TZI0012', 'Process Inlet Temperature', 'Degrees'],
    ['FI0010', 'Process Flow Rate', 'Mm^3/day'],
    ['TT0025', 'Coolant Inlet Temperature', 'Degrees'],
    ['TT0026', 'Coolant Outlet Temperature', 'Degrees'],
    ['PI0001', 'Coolant Inlet Pressure', 'Bar'],
    ['FI0027', 'Coolant Flow Rate', 'Mm^3/day'],
    ['TIC0022U', 'Coolant Valve Opening', '%'],
    ['PDT0024', 'Coolant Pressure Difference', 'Bar'],
]

# List of column names to ignore completely
irrelevantColumns = [
	'PI0001',
	'FI0027',
	'TIC0022U',
	'PDT0024',
    'PDI0064',
]

# List of column names used a targets
targetColumns = [
	'TT0026',
]

# List of training periods on form ['start', 'end']
traintime = [
	["2019-04-24 00:00:00", "2019-08-01 00:00:00"]
]

# Testing period, recommended: entire dataset
testtime = [
	"2017-01-01 00:00:00",
	"2020-03-01 00:00:00",
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