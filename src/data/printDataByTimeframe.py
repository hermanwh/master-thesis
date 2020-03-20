import pandas as pd
import time
import utilities
import sys

def main(filename, relevantColumns, start, end):
    start_time = time.time()
    utilities.printEmptyLine()

    print("Running", pyName)
    print("Prints the pandas dataframe")
    utilities.printHorizontalLine()

    df = utilities.readFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = utilities.getDataByTimeframe(df, start, end)

    if relevantColumns:
        df = utilities.dropIrrelevantColumns(df)

    utilities.printDataframe(df)

    try:
        print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    except NameError:
        print("Program finished, but took too long to count")
    utilities.printEmptyLine()

pyName = "printDataByTimeframe.py"
arguments = [
    "- file name (string)",
    "- relevantColumns (boolean)",
    "- start (string)",
    "- end (string)",
]

# usage: python ml/printDataByTimeframe.py datasets/filename.csv relevantColumns(bool) start end
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        relevantColumns = int(sys.argv[2])
        start = sys.argv[3]
        end = sys.argv[4]
    except:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()
    main(filename, relevantColumns, start, end)