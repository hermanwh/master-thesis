
import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import matplotlib.pyplot as plt
import pandas as pd
import time
import utilities
import plots
import metrics
from configs import getConfig

def main(filename, showRelevantColumns):
    subdir = filename.split('/')[-2].upper()
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    start_time = time.time()
    utilities.printEmptyLine()

    print("Running", pyName)
    print("Plots the pandas dataframe")
    utilities.printHorizontalLine()

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)

    if showRelevantColumns:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    plots.plotData(df, plt, columnDescriptions=labelNames, columnUnits=columnUnits)
    plt.show()

    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    utilities.printEmptyLine()

pyName = "plotData.py"
arguments = [
    "- file name (string)",
    "- relevantColumns (boolean)",
]

# usage: python ml/plotData.py datasets/filename.csv relevantColumns(bool)
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        relevantColumns = int(sys.argv[2])
    except:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()
    main(filename, relevantColumns)