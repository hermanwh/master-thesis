import pandas as pd
import sys
import time

def main(filename, targetfilename, start, end):
    start_time = time.time()

    print("Loading file {}".format(filename))
    df = pd.read_csv(filename)
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_copy.set_index('Date', inplace=True)
    df_copy.insert(loc=0, column='Date', value=df['Date'].values)
    df_copy = df_copy.loc[start:end]
    print("Writing file {}".format(targetfilename))
    df_copy.to_csv(targetfilename, index=False)

    print("Running of discardData.py finished in", time.time() - start_time, "seconds")

# usage: python discardData.py file targetfile start end
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        targetfilename = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
    except:
        print("discardData.py was called with inappropriate arguments")
        print("Please provide the following arguments:")
        print("- file name (string)")
        print("- target file name (string)")
        print("- start (string)")
        print("- end (string)")
        sys.exit()
    main(filename, targetfilename, start, end)