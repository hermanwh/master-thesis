# basic D_h

import src.core as mlModule

model = 'A'

# File path to dataset .csv file
filename = "../master-thesis-db/datasets/D/dataD.csv"

# List of columns on form ['name', 'description', 'unit']
columns = [
    ['20TT001', 'Process Inlet Temperature', 'Degrees'],
    ['20PT001', 'Process Inlet Pressure', 'Bar'],
    ['20FT001', 'Process Flow Rate', 'kg/hour'],
    ['20TT002', 'Process Outlet Temperature', 'Degrees'],
    ['20PDT001', 'Process Pressure Difference', 'Bar'],
    ['50TT001', 'Coolant Inlet Temperature', 'Degrees'],
    ['50PT001', 'Coolant Inlet Pressure', 'Bar'],
    ['50FT001', 'Coolant Flow Rate', 'kg/hour'],
    ['50TT002', 'Coolant Outlet Temperature', 'Degrees'],
    ['50PDT001', 'Coolant Pressure Difference', 'Bar'],
    ['50TV001', 'Coolant Valve Opening', '%'],
]

# List of column names to ignore completely
irrelevantColumns = [
    '20PT001',
    '50PT001',
    '50FT001',
    '50PDT001',
    '50TV001',
]

# List of column names used a targets
targetColumns = [
    '50TT002',
    '20PDT001',
]

# List of training periods on form ['start', 'end']
traintime = [
    ["2020-01-01 00:00:00", "2020-03-20 00:00:00"],
]

# Testing period, recommended: entire dataset
testtime = [
    "2020-01-01 00:00:00",
    "2020-08-01 00:00:00"
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

import src.utils.utilities as utils
import src.utils.plots as plots
import matplotlib.pyplot as plt

df_fouling = utils.readDataFile('../master-thesis-db/datasets/D/foulingC.csv')
df_fouling = utils.getDataWithTimeIndex(df_fouling)

def foulingPlot(deviationsList, name):
    y1 = deviationsList[0][0][2]
    y2 = (-1)* df_fouling['Fouling'].values[mlModule._maxEnrolWindow:]
    x = mlModule._indexColumn

    plots.duoPlot(y1, y2, x, plt, y2lim=(-0.000125+(min(y1)+2.5)*0.00005, 0.0+max(y1)*0.000055), textArgs=[
        'Deviation Coolant Outlet Temperature vs. Added fouling, ' + name,
        'Date',
        'Temperature deviation',
        'Added fouling',
    ])

for model in modelList:
    mlModule.initModels([model])
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
        plot=True,
        interpol=False,
        score=False,
    )
    foulingPlot(deviationsList, model.name)