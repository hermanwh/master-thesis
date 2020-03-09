import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import numpy as np
import utilities
import plots
from configs import getConfig
from sklearn.preprocessing import StandardScaler


def covmat(df, relevantColumns=None, columnDescriptions=None):
    if relevantColumns:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, columnDescriptions])

    if 'Date' in df.columns:
        df = df.drop('Date', axis=1, inplace=False)

    utilities.printColumns(df, columnDescriptions)
    
    x = df.values
    standardScaler = StandardScaler()
    x = standardScaler.fit_transform(x)
    covMat = np.cov(x.T)

    return covMat

def printCovMat(covMat):
    utilities.printHorizontalLine()
    print("Covariance matrix")
    utilities.prettyPrint(covMat, 2, True)
    utilities.printHorizontalLine()

def preCovmat(filename, colDescs):
    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    return covmat(df, columnDescriptions=colDescs)

pyName = "covmat.py"
arguments = [
    "- file name (string)",
]

# usage: python src/ml/analysis/covmat.py ../datasets/subdir/filename.csv
if __name__ == "__main__":
    start_time = time.time()
    utilities.printEmptyLine()
    
    print("Running", pyName)
    print("Calculates the correlation matrix of relevant dataset columns")
    utilities.printHorizontalLine()

    try:
        filename = sys.argv[1]
    except:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()

    subdir = filename.split('/')[1]
    columns, labelNames, units, relevantColumns, timestamps = getConfig(subdir)

    cov = preCovmat(filename, labelNames)

    printCovMat(cov)
    
    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    utilities.printEmptyLine()

