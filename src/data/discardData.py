import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import utilities
import prints
import plots
import pandas as pd
from configs import getConfig

def main(filename, targetfilename, start, end):
    print("Loading file {}".format(filename))
    df = pd.read_csv(filename)
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_copy.set_index('Date', inplace=True)
    df_copy.insert(loc=0, column='Date', value=df['Date'].values)
    df_copy = df_copy.loc[start:end]
    print("Writing file {}".format(targetfilename))
    df_copy.to_csv(targetfilename, index=False)

pyName = "discardData.py"
arguments = [
    "- filename (string)",
    "- target filename (string)",
    "- start time (string)",
    "- end time (string)",
]

# usage: python discardData.py file targetfile start end
if __name__ == "__main__":
    start_time = time.time()
    prints.printEmptyLine()
    
    print("Running", pyName)
    print("Prints dataframe")
    prints.printEmptyLine()
    
    try:
        filename = sys.argv[1]
        targetfilename = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
    except IndexError:
        print(pyName, "was called with inappropriate arguments")
        print("Please provide the following arguments:")
        for argument in arguments:
            print(argument)
        sys.exit()

    main(filename, targetfilename, start, end)

    prints.printEmptyLine()
    print("Running of", pyName, "finished in", time.time() - start_time, "seconds")
    prints.printEmptyLine()