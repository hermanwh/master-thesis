import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import numpy as np
import utilities
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configs import getConfig
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

def pcaPlot(filename):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)
    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    traintime, testtime, validtime = timestamps

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    start_train, end_train = traintime
    start_test, end_test = testtime
    start_valid, end_valid = validtime

    df_train = utilities.getDataByTimeframe(df, start_train, end_train)
    train_vals = df_train.values
    #train_vals = df.values

    sc = StandardScaler()
    train_vals = sc.fit_transform(train_vals)
    
    numberOfComponents = 2

    pca = decomposition.PCA(n_components=numberOfComponents)
    pca.fit(train_vals)

    x = df.values
    x = sc.transform(x)
    x = pca.transform(x)

    df_pca = pd.DataFrame(data = x, index=df.index, columns=['pca1', 'pca2'])
    df_pca_train = utilities.getDataByTimeframe(df_pca, start_train, end_train)
    df_pca_test = utilities.getDataByTimeframe(df_pca, end_train, end_test)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PCA 1', fontsize=10)
    ax.set_ylabel('PCA 2', fontsize=10)
    ax.set_title('PCA plot', fontsize=12)
    cmap = sns.cubehelix_palette(as_cmap=True)
    indexx = list(range(df_pca_test.shape[0]))
    ax.scatter(df_pca_train['pca1'], df_pca_train['pca2'], c = 'lightblue')
    points = ax.scatter(df_pca_test['pca1'], df_pca_test['pca2'], c = indexx, cmap = cmap, alpha=0.4)
    fig.colorbar(points)
    plt.show()

    return pca

def printReconstructionRow(pca, x, standardScaler):
    transformed = pca.transform(x)
    inv_transformed = pca.inverse_transform(transformed)
    inv_standardized = standardScaler.inverse_transform(inv_transformed)

    print("Top row before standardization and PCA")
    print(np.array_str(x[:1,:], precision=2, suppress_small=True))
    utilities.printHorizontalLine()

    print("Top row after reconstruction")
    print(np.array_str(inv_standardized[:1,:], precision=2, suppress_small=True))
    utilities.printHorizontalLine()

def printExplainedVarianceRatio(pca):
    utilities.printHorizontalLine()
    print("Variance ratio explained by each principal component")
    utilities.prettyPrint(pca.explained_variance_ratio_, 2, True)
    utilities.printHorizontalLine()

pyName = "pcaPlot.py"
arguments = [
    "- filename (string)",
]

# usage: python src/ml/analysis/pca.py datasets/subdir/filename.csv
if __name__ == "__main__":
    start_time = time.time()
    utilities.printEmptyLine()
    
    print("Running", pyName)
    print("Performs Principal Component Analysis on relevant dataset columns")
    utilities.printHorizontalLine()

    try:
        filename = sys.argv[1]
    except IndexError:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()

    pca = pcaPlot(filename)

    printExplainedVarianceRatio(pca)

    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    utilities.printEmptyLine()