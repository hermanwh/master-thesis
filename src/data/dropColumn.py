import pandas as pd
import sys
import time

def main(filename, column, target_file):
    start_time = time.time()

    print("Loading file {}".format(filename))
    df_iris = pd.read_csv(filename).drop(column, axis=1)
    print("Writing file {}".format(target_file))
    df_iris.to_csv(target_file, index=False)

    print("Running of dropColumn.py finished in", time.time() - start_time, "seconds")

# usage: python dropColumn.py file targetfile column
if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        target_file = sys.argv[2]
        column = sys.argv[3]
    except:
        print("discardData.py was called with inappropriate arguments")
        print("Please provide the following arguments:")
        print("- file name (string)")
        print("- target file name (string)")
        print("- start (string)")
        print("- end (string)")
        sys.exit()
    main(filename, column, target_file)