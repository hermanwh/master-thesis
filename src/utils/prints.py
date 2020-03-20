import numpy as np
import utilities

from prettytable import PrettyTable

def printModelScores(names, r2_train, r2_test):
    print("Model prediction scores")
    t = PrettyTable(['Model', 'Train score', 'Test score'])
    for i, name in enumerate(names):
        t.add_row([name, round(r2_train[i], 4), round(r2_test[i], 4)])
    print(t)

def printDataframe(df):
    print(df)
    printHorizontalLine()

def printDataframeByTimeframe(df, start, end):
    df = utilities.getDataByTimeframe(df, start, end)
    printDataframe(df)

def printColumns(df, columnDescriptions):
    printHorizontalLine()
    print("Dataset columns:")
    for i, column in enumerate(df.columns):
        if columnDescriptions is not None and column in columnDescriptions:
            print("Col.", i, ":", column, "-", columnDescriptions[column])
        else:
            print("Col.", i, ":", column)

def prettyPrint(data, precision, suppress):
    print(np.array_str(data, precision=precision, suppress_small=suppress))

def printEmptyLine():
    print("")

def printHorizontalLine():
    print("-------------------------------------------")
