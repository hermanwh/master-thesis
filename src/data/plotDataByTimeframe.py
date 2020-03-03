
import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
import pandas as pd
import time
import utilities
import sys
import matplotlib.pyplot as plt
from configs import getConfig

def main(filename, showRelevantColumns, start, end):
    subdir = filename.split('/')[-2].upper()
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    print("Running", pyName)
    print("Plots the pandas dataframe")
    utilities.printHorizontalLine()

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = utilities.getDataByTimeframe(df, start, end)

    if showRelevantColumns:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    utilities.plotData(df, plt, columnDescriptions=labelNames)
    plt.show()

    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    utilities.printEmptyLine()

pyName = "plotDataByTimeframe.py"
arguments = [
    "- file name (string)",
    "- relevantColumns (boolean)",
    "- start (string)",
    "- end (string)",
]

# usage: python ml/plotDataByTimeframe.py datasets/filename.csv relevantColumns(bool) start end
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        relevantColumns = int(sys.argv[2])
        start = pd.to_datetime(sys.argv[3])
        end = pd.to_datetime(sys.argv[4])
    except:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()
    main(filename, relevantColumns, start, end)