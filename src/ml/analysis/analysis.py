import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import utilities
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
import pandas as pd
import covmat
from configs import getConfig
from sklearn.preprocessing import MinMaxScaler, StandardScaler

colors = list(utilities.getColorScheme().values())
scaler = StandardScaler()

sea.set(context='paper', style='whitegrid', palette=sea.color_palette(colors))
plt.style.use(ROOT_PATH + '/src/utils/matplotlib_params.rc')

def columnsPlot(df_train, df_test):
    fig, axs = plt.subplots(nrows=df_train.shape[-1], ncols=2, figsize=(15,20), dpi=100)
    #fig.tight_layout()
    
    for k in range(df_train.shape[-1]):
        ax1, ax2 = axs[k, 0], axs[k, 1]
        
        ax1.plot(df_train.iloc[:,k], label="train",
                marker="o", ms=.8, lw=0)
        ax1.plot(df_test.iloc[:,k], label="valid",
                marker="o", ms=.8, lw=0)
        
        ax1.set_xticks(ax1.get_xticks()[3::3])
        ax1.set_ylabel(df_train.columns[k])
        
        sea.distplot(df_train.iloc[:,k], ax=ax2, label="train", kde=True, kde_kws={"lw":2.5})
        sea.distplot(df_test.iloc[:,k], ax=ax2, label="test", kde=True, kde_kws={"lw":2.5})
        
        ax2.set_xlim((-3,3))
        ax2.legend(loc="upper right")

    fig.show()

    input("Press any key to close")

def targetsPlot(df_train, df_test):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,6), dpi=100, sharex=True, sharey=True)

    for k in range(2):
        ax = axs[k]
        
        sea.distplot(df_train.iloc[:,k], ax=ax, kde=True, kde_kws={"lw":2.5}, label="train")
        sea.distplot(df_test.iloc[:,k], ax=ax, kde=True, kde_kws={"lw":2.5}, label="test")
        
        ax.set_xlim((-5,5))
        ax.legend(frameon=True, loc='upper left')
        
    fig.show()

    input("Press any key to close")

def scatterPlot(df_train):
    plt.figure()
    sea.pairplot(df_train, vars=df_train.columns, diag_kind="kde")
    plt.show()

    input("Press any key to close")

def correlationPlot(df, title="Correlation plot"):
    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15,20), dpi=100)

    # Generate a custom diverging colormap
    cmap = sea.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sea.heatmap(corr, mask=mask, cmap=cmap,
                square=True, linewidths=1, cbar_kws={"shrink": .6})

    ax.set_title(title)
    
    f.show()

    input("Press any key to close")

def main(filename):
    subdir = filename.split('/')[-2]
    columns, relevantColumns, labelNames, columnUnits, timestamps = getConfig(subdir)

    df = utilities.readDataFile(filename)
    df = utilities.getDataWithTimeIndex(df)
    df = df.dropna()

    traintime, testtime, validtime = timestamps

    if relevantColumns is not None:
        df = utilities.dropIrrelevantColumns(df, [relevantColumns, labelNames])

    start_train, end_train = traintime
    start_test, end_test = testtime

    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=list(map((lambda x: x.split('_')[0]), df.columns)))

    #scaled_df = scaled_df.copy().resample('H').mean()
    #scaled_df = scaled_df.dropna()

    df_train = utilities.getDataByTimeframe(scaled_df, start_train, end_train)
    df_test = utilities.getDataByTimeframe(scaled_df, start_test, end_test)

    columnsPlot(df_train, df_test)
    #targetsPlot(df_train, df_test)
    
    cov = covmat.covmat(df_train)
    covmat.printCovMat(cov)

    correlationPlot(df_train)
    #scatterPlot(df_train)


    df_train = df_train.loc[:,~df_train.columns.duplicated()]
    #print(df_train)

    pd.plotting.scatter_matrix(df_train, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

    #asd = df_train.copy().resample('H').mean()
    #print(asd)
    #sea.pairplot(asd, vars=df_train.columns)
    input("Press any key to close")

# usage: python ml/covmat.py datasets/filename.csv
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)
