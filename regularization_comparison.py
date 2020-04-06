import src.core as mlApi
import src.core_configs as configs

# 1. Define dataset specifics
filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('D', 'A', '30min')

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlpr_1_1 = mlApi.MLP('MLPr 1x64 1.0', layers=[64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr_1_2 = mlApi.MLP('MLPr 1x64 0.5', layers=[64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr_1_3 = mlApi.MLP('MLPr 1x64 0.1', layers=[64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr_1_4 = mlApi.MLP('MLPr 1x64 0.05', layers=[64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr_1_5 = mlApi.MLP('MLPr 1x64 0.01', layers=[64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr_1_6 = mlApi.MLP('MLPr 1x64 0.005', layers=[64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr_1_7 = mlApi.MLP('MLPr 1x64 0.001', layers=[64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd_1_8 = mlApi.MLP('MLPd 1x64 0.2', layers=[64], dropout=0.2, epochs=5000)

mlpr_2_1 = mlApi.MLP('MLPr 2x64 1.0', layers=[64, 64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr_2_2 = mlApi.MLP('MLPr 2x64 0.5', layers=[64, 64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr_2_3 = mlApi.MLP('MLPr 2x64 0.1', layers=[64, 64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr_2_4 = mlApi.MLP('MLPr 2x64 0.05', layers=[64, 64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr_2_5 = mlApi.MLP('MLPr 2x64 0.01', layers=[64, 64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr_2_6 = mlApi.MLP('MLPr 2x64 0.005', layers=[64, 64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr_2_7 = mlApi.MLP('MLPr 2x64 0.001', layers=[64, 64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd_2_8 = mlApi.MLP('MLPd 2x64 0.2', layers=[64, 64], dropout=0.2, epochs=5000)

linear_r = mlApi.Linear_Regularized('linear')

modelLists = [
    [
        mlpr_1_1, mlpr_1_2, mlpr_1_3, mlpr_1_4, mlpr_1_5, mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
    ],
    [
        mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
    ],
    [
        mlpr_2_1, mlpr_2_2, mlpr_2_3, mlpr_2_4, mlpr_2_5, mlpr_2_6, mlpr_2_7, mlpd_2_8, linear_r,
    ],
    [
        mlpr_2_6, mlpr_2_7, mlpd_2_8, linear_r,
    ]
]

for modelList in modelLists:
    # 4. Initiate and train models
    mlApi.initModels(modelList)
    retrain=False
    mlApi.trainModels(retrain)

    # 5. Predict
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
        plot=True,
        interpol=False,
    )

mlApi.reset()

# 1. Define dataset specifics
filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('F', 'A', '30min')

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlpr_1_1 = mlApi.MLP('MLPr 1x64 1.0', layers=[64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr_1_2 = mlApi.MLP('MLPr 1x64 0.5', layers=[64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr_1_3 = mlApi.MLP('MLPr 1x64 0.1', layers=[64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr_1_4 = mlApi.MLP('MLPr 1x64 0.05', layers=[64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr_1_5 = mlApi.MLP('MLPr 1x64 0.01', layers=[64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr_1_6 = mlApi.MLP('MLPr 1x64 0.005', layers=[64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr_1_7 = mlApi.MLP('MLPr 1x64 0.001', layers=[64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd_1_8 = mlApi.MLP('MLPd 1x64 0.2', layers=[64], dropout=0.2, epochs=5000)

mlpr_2_1 = mlApi.MLP('MLPr 2x64 1.0', layers=[64, 64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr_2_2 = mlApi.MLP('MLPr 2x64 0.5', layers=[64, 64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr_2_3 = mlApi.MLP('MLPr 2x64 0.1', layers=[64, 64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr_2_4 = mlApi.MLP('MLPr 2x64 0.05', layers=[64, 64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr_2_5 = mlApi.MLP('MLPr 2x64 0.01', layers=[64, 64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr_2_6 = mlApi.MLP('MLPr 2x64 0.005', layers=[64, 64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr_2_7 = mlApi.MLP('MLPr 2x64 0.001', layers=[64, 64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd_2_8 = mlApi.MLP('MLPd 2x64 0.2', layers=[64, 64], dropout=0.2, epochs=5000)

linear_r = mlApi.Linear_Regularized('linear')

modelLists = [
    [
        mlpr_1_1, mlpr_1_2, mlpr_1_3, mlpr_1_4, mlpr_1_5, mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
    ],
    [
        mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
    ],
    [
        mlpr_2_1, mlpr_2_2, mlpr_2_3, mlpr_2_4, mlpr_2_5, mlpr_2_6, mlpr_2_7, mlpd_2_8, linear_r,
    ],
    [
        mlpr_2_6, mlpr_2_7, mlpd_2_8, linear_r,
    ]
]

for modelList in modelLists:
    # 4. Initiate and train models
    mlApi.initModels(modelList)
    retrain=False
    mlApi.trainModels(retrain)

    # 5. Predict
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
        plot=True,
        interpol=False,
    )

mlApi.reset()

# 1. Define dataset specifics
filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig('G', 'A', '30min')

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlpr_1_1 = mlApi.MLP('MLPr 1x64 1.0', layers=[64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr_1_2 = mlApi.MLP('MLPr 1x64 0.5', layers=[64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr_1_3 = mlApi.MLP('MLPr 1x64 0.1', layers=[64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr_1_4 = mlApi.MLP('MLPr 1x64 0.05', layers=[64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr_1_5 = mlApi.MLP('MLPr 1x64 0.01', layers=[64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr_1_6 = mlApi.MLP('MLPr 1x64 0.005', layers=[64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr_1_7 = mlApi.MLP('MLPr 1x64 0.001', layers=[64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd_1_8 = mlApi.MLP('MLPd 1x64 0.2', layers=[64], dropout=0.2, epochs=5000)

mlpr_2_1 = mlApi.MLP('MLPr 2x64 1.0', layers=[64, 64], l1_rate=1.0, l2_rate=1.0, epochs=5000)
mlpr_2_2 = mlApi.MLP('MLPr 2x64 0.5', layers=[64, 64], l1_rate=0.5, l2_rate=0.5, epochs=5000)
mlpr_2_3 = mlApi.MLP('MLPr 2x64 0.1', layers=[64, 64], l1_rate=0.1, l2_rate=0.1, epochs=5000)
mlpr_2_4 = mlApi.MLP('MLPr 2x64 0.05', layers=[64, 64], l1_rate=0.05, l2_rate=0.05, epochs=5000)
mlpr_2_5 = mlApi.MLP('MLPr 2x64 0.01', layers=[64, 64], l1_rate=0.01, l2_rate=0.01, epochs=5000)
mlpr_2_6 = mlApi.MLP('MLPr 2x64 0.005', layers=[64, 64], l1_rate=0.005, l2_rate=0.005, epochs=5000)
mlpr_2_7 = mlApi.MLP('MLPr 2x64 0.001', layers=[64, 64], l1_rate=0.001, l2_rate=0.001, epochs=5000)
mlpd_2_8 = mlApi.MLP('MLPd 2x64 0.2', layers=[64, 64], dropout=0.2, epochs=5000)

linear_r = mlApi.Linear_Regularized('linear')

modelLists = [
    [
        mlpr_1_1, mlpr_1_2, mlpr_1_3, mlpr_1_4, mlpr_1_5, mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
    ],
    [
        mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
    ],
    [
        mlpr_2_1, mlpr_2_2, mlpr_2_3, mlpr_2_4, mlpr_2_5, mlpr_2_6, mlpr_2_7, mlpd_2_8, linear_r,
    ],
    [
        mlpr_2_6, mlpr_2_7, mlpd_2_8, linear_r,
    ]
]

for modelList in modelLists:
    # 4. Initiate and train models
    mlApi.initModels(modelList)
    retrain=False
    mlApi.trainModels(retrain)

    # 5. Predict
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
        plot=True,
        interpol=False,
    )