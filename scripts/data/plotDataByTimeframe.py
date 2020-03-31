import sys
import os

import time
import ..utils.utilities
import ..utils.prints
import ..utils.plots
import matplotlib.pyplot as plt
from ..utils.configs import getConfig

def main(filename, start, end):
    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()
    df = utilities.getDataByTimeframe(df, start, end)

    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    plots.plotData(df, plt, columnDescriptions=labelNames)
    plt.show()

pyName = "plotDataByTimeframe.py"
arguments = [
    "- file name (string)",
    "- start (string)",
    "- end (string)",
]

# usage: python ml/plotDataByTimeframe.py datasets/filename.csv relevantColumns(bool) start end
if __name__ == "__main__":
    start_time = time.time()
    prints.printEmptyLine()
    
    print("Running", pyName)
    print("Plots dataframe by provided timeframe")
    prints.printEmptyLine()
    
    try:
        filename = sys.argv[1]
        start = sys.argv[2]
        end = sys.argv[3]
    except IndexError:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()

    main(filename, start, end)

    prints.printEmptyLine()
    print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    prints.printEmptyLine()