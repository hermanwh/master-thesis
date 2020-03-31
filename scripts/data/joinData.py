import sys
import os

import time
import ..utils.utilities
import ..utils.prints
import ..utils.plots
import pandas as pd
from ..utils.configs import getConfig

def main(targetfile, files):
    file1 = files[0]
    print("Loading file {}".format(file1))
    df = pd.read_csv(file1)
    print("- Shape of file", file1, ":", df.shape)
    rem_files = files[1:]

    for fileName in rem_files:
        print("Loading file {}".format(fileName))
        tempdf = pd.read_csv(fileName)
        print("- Shape of file", fileName, ":", tempdf.shape)
        df = df.append(tempdf)

    print("Total file shape:", df.shape)

    print("Writing file {}".format(targetfile))
    df.to_csv(targetfile, index=False)

pyName = "joinData.py"
arguments = [
    "- target file name (string)",
    "- filename 1",
    "- filename 2",
    "- ...",
]

# usage: python joinData.py targetfile.csv file1.csv file2.csv ...
if __name__ == "__main__":
    start_time = time.time()
    prints.printEmptyLine()

    print("Running", pyName)
    print("Prints dataframe")
    prints.printEmptyLine()

    try:
        targetfile = sys.argv[1]
        testing = sys.argv[3]
        files = sys.argv[2:]
    except IndexError:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()

    main(targetfile, files)

    prints.printEmptyLine()
    print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    prints.printEmptyLine()