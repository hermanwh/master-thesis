from covmat import (covmat, printCovMat)
from pca import (pca, printExplainedVarianceRatio)
import utilities
import plots
import metrics
import matplotlib.pyplot as plt
import sys

from config.dataConfigs import (getConfig)

def main(filename):
    subdir = filename.split('/')[1]
    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    #df = df[df['FT0005'] != 0.0]
    print(df)
    low = .05
    high = 0.95
    print(df[df.index.duplicated()])
    df = df.loc[~df.index.duplicated(keep='first')]
    quant_df = df.quantile([low, high])
    df = df.apply(lambda x: x[(x >= quant_df.loc[low,x.name]) & (x <= quant_df.loc[high,x.name])], axis=0)
    df = df.dropna()
    print(df)
    df.to_csv('processed.csv', index=True)
    #start, end = traintime

    #df_train = utilities.getDataByTimeframe(df, start, end)
    if relevantColumns is not  None:
        df = dropIrrelevantColumns(df, [relevantColumns, labelNames])

    if time is not None:
        traintime, testtime, validtime = time
        df = utilities.getDataByTimeframe(df, traintime[0], traintime[1])

    print(df)

    cov = covmat(df, None, labelNames)
    printCovMat(cov)

    prints.printEmptyLine()

    pca_calc = pca(df, 5, None, labelNames)
    printExplainedVarianceRatio(pca_calc)


    plots.plotData(df, plt, labelNames)
    plt.show()

# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)

