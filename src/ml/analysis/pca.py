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
from configs import getConfig
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

def pca(df, numberOfComponents, relevantColumns=None, columnDescriptions=None):
    if relevantColumns:
        df = utilities.dropIrrelevantColumns(df, {relevantColumns, columnDescriptions})

    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)

    prints.printColumns(df, columnDescriptions)

    x = df.values
    standardScaler = StandardScaler()
    x = standardScaler.fit_transform(x)

    if numberOfComponents == 0:
        numberOfComponents = df.shape[1]

    pca = decomposition.PCA(n_components=numberOfComponents)
    pca.fit(x)

    return pca

def printReconstructionRow(pca, x, standardScaler):
    transformed = pca.transform(x)
    inv_transformed = pca.inverse_transform(transformed)
    inv_standardized = standardScaler.inverse_transform(inv_transformed)

    print("Top row before standardization and PCA")
    print(np.array_str(x[:1,:], precision=2, suppress_small=True))
    prints.printHorizontalLine()

    print("Top row after reconstruction")
    print(np.array_str(inv_standardized[:1,:], precision=2, suppress_small=True))
    prints.printHorizontalLine()

def printExplainedVarianceRatio(pca):
    prints.printHorizontalLine()
    print("Variance ratio explained by each principal component")
    utilities.prettyPrint(pca.explained_variance_ratio_, 2, True)
    prints.printHorizontalLine()
    
def prePCA(filename, numberOfComponents):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()
        
    return pca(df, numberOfComponents, columnDescriptions=labelNames)

pyName = "pca.py"
arguments = [
    "- file name (string)",
    "- number of components (int)",
]

# usage: python src/ml/analysis/pca.py datasets/subdir/filename.csv nrOfComponents
if __name__ == "__main__":
    start_time = time.time()
    prints.printEmptyLine()
    
    print("Running", pyName)
    print("Performs Principal Component Analysis on relevant dataset columns")
    prints.printHorizontalLine()

    try:
        filename = sys.argv[1]
        numberOfComponents = int(sys.argv[2])
        if numberOfComponents < 1:
            numberOfComponents = 0
    except IndexError:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()

    pca = prePCA(filename, numberOfComponents)

    printExplainedVarianceRatio(pca)

    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    prints.printEmptyLine()