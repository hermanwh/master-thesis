# Regluarization comparison for multilayer perceptron networks

# %load regularization_comparison.py
import src.core as mlModule
import src.core_configs as configs

def initTrainPredict(modelList, retrain=False, plot=True, interpol=False):
    # 4. Initiate and train models
    mlModule.initModels(modelList)
    mlModule.trainModels(retrain)
    
    # 5. Predict
    modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlModule.predictWithModels(
        plot=plot,
        interpol=interpol,
    )
    
def pred(facility, model, resolution):
    filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder = configs.getConfig(facility, model, resolution)

    df = mlModule.initDataframe(filename, columns, irrelevantColumns)
    df_train, df_test = mlModule.getTestTrainSplit(traintime, testtime)
    X_train, y_train, X_test, y_test = mlModule.getFeatureTargetSplit(targetColumns)

    mlpr_1_1 = mlModule.MLP('MLP 1x128 1.0'+' mod'+model, layers=[128], l1_rate=1.0, epochs=5000)
    mlpr_1_2 = mlModule.MLP('MLP 1x128 0.5'+' mod'+model, layers=[128], l1_rate=0.5, epochs=5000)
    mlpr_1_3 = mlModule.MLP('MLP 1x128 0.1'+' mod'+model, layers=[128], l1_rate=0.1, epochs=5000)
    mlpr_1_4 = mlModule.MLP('MLP 1x128 0.05'+' mod'+model, layers=[128], l1_rate=0.05, epochs=5000)
    mlpr_1_5 = mlModule.MLP('MLP 1x128 0.01'+' mod'+model, layers=[128], l1_rate=0.01, epochs=5000)
    mlpr_1_6 = mlModule.MLP('MLP 1x128 0.005'+' mod'+model, layers=[128], l1_rate=0.005, epochs=5000)
    mlpr_1_7 = mlModule.MLP('MLP 1x128 0.001'+' mod'+model, layers=[128], l1_rate=0.001, epochs=5000)
    mlpd_1_8 = mlModule.MLP('MLP 1x128 0.2'+' mod'+model, layers=[128], dropout=0.2, epochs=5000)

    linear_r = mlModule.Linear_Regularized('Linear rCV'+' mod'+model)

    initTrainPredict([
            mlpr_1_1, mlpr_1_2, mlpr_1_3, mlpr_1_4, mlpr_1_5, mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
        ])

    initTrainPredict([
            mlpr_1_6, mlpr_1_7, mlpd_1_8, linear_r,
        ])
 
pred('D', 'A', '30min')
mlModule.reset()
pred('D', 'B', '30min')
mlModule.reset()
pred('F', 'A', '30min')
mlModule.reset()
pred('F', 'B', '30min')
mlModule.reset()
pred('G', 'A', '30min')
mlModule.reset()
pred('G', 'B', '30min')
mlModule.reset()
