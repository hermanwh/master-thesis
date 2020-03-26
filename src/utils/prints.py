from prettytable import PrettyTable

import numpy as np
import utilities

np.random.seed(100)

def printCorrelationMatrix(covMat, df, columnNames=None):
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1, inplace=False)
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1, inplace=False)

    for i, column in enumerate(df.columns):
        print(str(i) + " " + column + " " + columnNames[column]) if columnNames is not None and columnNames[column] else print(str(i) + " " + column)
    print("")
    prettyPrint(covMat, 2, True)

def printExplainedVarianceRatio(pca):
    print("Variance ratio explained by each principal component")
    prettyPrint(pca.explained_variance_ratio_, 2, True)
    
def printReconstructionRow(pca, x, standardScaler, index=0):
    transformed = pca.transform(x)
    inv_transformed = pca.inverse_transform(transformed)
    inv_standardized = standardScaler.inverse_transform(inv_transformed)

    print("Top row before standardization and PCA")
    prettyPrint(x[index,:], precision=2, suppress_small=True)
    
    print("Top row after reconstruction")
    prettyPrint(inv_standardized[index,:], precision=2, suppress_small=True)
    
def printModelScores(names, r2_train, r2_test):
    print("Model prediction scores")
    t = PrettyTable(['Model', 'Train score', 'Test score'])
    for i, name in enumerate(names):
        t.add_row([name, round(r2_train[i], 4), round(r2_test[i], 4)])
    print(t)

def printDataframe(df):
    print(df)

def printDataframeByTimeframe(df, start, end):
    df = utilities.getDataByTimeframe(df, start, end)
    printDataframe(df)

def printColumns(df, columnDescriptions):
    for i, column in enumerate(df.columns):
        if columnDescriptions is not None and column in columnDescriptions:
            print("Col.", i, ":", column, "-", columnDescriptions[column])
        else:
            print("Col.", i, ":", column)

def prettyPrint(data, precision, suppress):
    print(np.array_str(data, precision=precision, suppress_small=suppress))

def printEmptyLine():
    print("")

def printHorizontalLine():
    print("-------------------------------------------")

def printTrainingSummary(trainingSummary):
    t = PrettyTable(['Model', 'Min. loss', 'Chosen loss','Min. val loss', 'Epochs'])
    for name, summary in trainingSummary.items():
        t.add_row([name, round(summary['loss_final'], 6), round(summary['loss_actual'], 6), round(summary['val_loss_final'], 6), summary['length']])
    print(t)