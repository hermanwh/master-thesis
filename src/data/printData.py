import pandas as pd
import time
import utilities
import sys

def printData(df):
    print(df)
    utilities.printHorizontalLine()

def main(filename, relevantColumns):
    start_time = time.time()
    utilities.printEmptyLine()

    print("Running", pyName)
    print("Prints the pandas dataframe")
    utilities.printHorizontalLine()

    df = utilities.readFile(filename)
    df = utilities.getDataWithTimeIndex(df)

    if relevantColumns:
        df = utilities.dropIrrelevantColumns(df)

    printData(df)

    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    utilities.printEmptyLine()

pyName = "printData.py"
arguments = [
    "- file name (string)",
    "- relevantColumns (boolean)",
]

# usage: python ml/printData.py datasets/filename.csv relevantColumns(bool)
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