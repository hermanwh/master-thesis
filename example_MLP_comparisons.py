import src.utils.utilities as utilities
import src.utils.models as models
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


args = utilities.Args({
    'activation': 'relu',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'metrics': ['mean_squared_error'],
    'epochs': 2000,
    'batchSize': 32,
    'verbose': 0,
    'callbacks': utilities.getBasicCallbacks(),
    'enrolWindow': 0,
    'validationSize': 0.2,
    'testSize': 0.2,
})


names = list(map(lambda el: el[0], columns))
descriptions = list(map(lambda el: el[1], columns))
units = list(map(lambda el: el[2], columns))

relevantColumns = list(filter(lambda col: col not in irrelevantColumns, map(lambda el: el[0], columns)))
columnUnits = dict(zip(names, units))
columnDescriptions = dict(zip(names, descriptions))

df = mlApi.initDataframe(filename, relevantColumns, columnDescriptions)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlp_10 = models.kerasSequentialRegressionModel(
    params={
        'name': '10 normal',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [10, args.activation]
    ],
)

mlp_20 = models.kerasSequentialRegressionModel(
    params={
        'name': '20 normal',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [20, args.activation]
    ],
)

mlp_128 = models.kerasSequentialRegressionModel(
    params={
        'name': '128 normal',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [128, args.activation]
    ],
)

mlp_10_reg = models.kerasSequentialRegressionModelWithRegularization(
    params={
        'name': '10 reg',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [10, args.activation]
    ],
)

mlp_20_reg = models.kerasSequentialRegressionModelWithRegularization(
    params={
        'name': '20 reg',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [20, args.activation]
    ],
)

mlp_128_reg = models.kerasSequentialRegressionModelWithRegularization(
    params={
        'name': '128 reg',
        'X_train': X_train,
        'y_train': y_train,
        'args': args,
    },
    structure=[
        [128, args.activation]
    ],
)

sklearnLinearModel = models.sklearnRidgeCV(
    params={
        'name': 'Linear',
        'X_train': X_train,
        'y_train': y_train,
    },
) 

modelList = [
    mlp_10,
    mlp_20,
    mlp_128,
    mlp_10_reg,
    mlp_20_reg,
    mlp_128_reg,
    sklearnLinearModel,
]

mlApi.initModels(modelList)
retrain=True
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test = mlApi.predictWithModels(plot=True)
