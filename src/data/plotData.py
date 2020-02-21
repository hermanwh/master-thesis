import pandas as pd
import time
import utilities
import sys
import matplotlib.pyplot as plt

def plotData(df):
    for column in df.columns:
        if column != "Date":
            fig, ax1 = plt.subplots()
            ax1.set_title('Plot of dataset column ' + column)
            color = 'darkgreen'
            ax1.set_xlabel('Date')
            ax1.set_ylabel(utilities.getColumnDescriptionOfColumn(column), color=color)
            ax1.plot(df.index, df[column], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(1, axis='y')
    plt.show()

def main(filename, relevantColumns):
    start_time = time.time()
    utilities.printEmptyLine()

    print("Running", pyName)
    print("Plots the pandas dataframe")
    utilities.printHorizontalLine()

    df = utilities.readFile(filename)
    df = utilities.getDataWithTimeIndex(df)

    if relevantColumns:
        df = utilities.dropIrrelevantColumns(df)
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    plotData(df)

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