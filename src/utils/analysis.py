import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import numpy as np
import utilities
import plots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configs import getConfig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

colors = list(utilities.getColorScheme().values())

sns.set(context='paper', style='whitegrid', palette=sns.color_palette(colors))
plt.style.use(ROOT_PATH + '/src/utils/matplotlib_params.rc')

def correlationMatrix(df):
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1, inplace=False)
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1, inplace=False)

    X = df.values
    standardScaler = StandardScaler()
    X = standardScaler.fit_transform(X)
    covMat = np.cov(X.T)

    return covMat

def pca(df, numberOfComponents, relevantColumns=None, columnDescriptions=None):
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1, inplace=False)
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1, inplace=False)

    X = df.values
    standardScaler = StandardScaler()
    X = standardScaler.fit_transform(X)

    if numberOfComponents < 1 or numberOfComponents > df.shape[1]:
        numberOfComponents = df.shape[1]

    pca = PCA(n_components=numberOfComponents)
    pca.fit(X)

    return pca

def pcaPlot(df, timestamps=None):
    if timestamps is not None:
        traintime, testtime, validtime = timestamps
        df_train, df_test = utilities.getTestTrainSplit(df, traintime, testtime)
        train_vals = df_train.values
    else:
        train_vals = df.values

    sc = StandardScaler()
    train_vals = sc.fit_transform(train_vals)
    
    numberOfComponents = 2

    pca = PCA(n_components=numberOfComponents)
    pca.fit(train_vals)

    X = df.values
    X = sc.transform(X)
    X = pca.transform(X)

    df_pca = pd.DataFrame(data = X, index=df.index, columns=['pca1', 'pca2'])
    if timestamps is not None:
        df_pca_train, df_pca_test = utilities.getTestTrainSplit(df_pca, traintime, testtime)
    else:
        df_pca_train, df_pca_test = None, df_pca

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PCA 1', fontsize=10)
    ax.set_ylabel('PCA 2', fontsize=10)
    ax.set_title('PCA plot', fontsize=12)
    cmap = sns.cubehelix_palette(as_cmap=True)
    indexx = list(range(df_pca_test.shape[0]))
    if df_pca_train is not None:
        ax.scatter(df_pca_train['pca1'], df_pca_train['pca2'], c = 'red')
    points = ax.scatter(df_pca_test['pca1'], df_pca_test['pca2'], c = indexx, cmap = cmap, alpha=0.2)
    fig.colorbar(points)
    plt.show()

def pairplot(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    
    if scaled_df.shape[0] > 1000:
        scaled_df = scaled_df.resample('H').mean()
    sns.pairplot(scaled_df, vars=scaled_df.columns, height=1.1)
    plt.show()

def scatterplot(df):
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

def correlationPlot(df, title="Correlation plot"):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    corr = scaled_df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8,6), dpi=100)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,
                square=True, linewidths=1, cbar_kws={"shrink": .6})

    ax.set_title(title)
    
    plt.show()

def valueDistribution(df, traintime, testtime):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    df_train, df_test = utilities.getTestTrainSplit(scaled_df, traintime, testtime)    

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
        
        sns.distplot(df_train.iloc[:,k], ax=ax2, label="train", kde=True, kde_kws={"lw":2.5})
        sns.distplot(df_test.iloc[:,k], ax=ax2, label="test", kde=True, kde_kws={"lw":2.5})
        
        ax2.set_xlim((-3,3))
        ax2.legend(loc="upper right")

    plt.show()