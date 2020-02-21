import pandas as pd
import sys
import time

def main(targetfile, files):
    start_time = time.time()

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

    print("Writing file {}".format(target_file))
    df.to_csv(target_file, index=False)

    print("Running of joinData.py finished in", time.time() - start_time, "seconds")


# usage: python joinData.py targetfile.csv file1.csv file2.csv ...
if __name__ == "__main__":
    try:
        targetfile = sys.argv[1]
        testing = sys.argv[3]
        files = sys.argv[2:]
    except:
        print("joinData.py was called with inappropriate arguments")
        print("Please provide the following arguments:")
        print("- target file name (string)")
        print("- file_nr_1 (string)")
        print("- file_nr_2 (string)")
        print("- ...")
        sys.exit()
    main(targetfile, files)