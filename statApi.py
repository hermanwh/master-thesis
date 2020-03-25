import sys
import os

ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt

import utilities
import metrics
import models
import modelFuncs
import plots
import prints
import analysis

def correlationMatrix(df):
    return analysis.correlationMatrix(df)

def pca(df, numberOfComponents, relevantColumns=None, columnDescriptions=None):
    return analysis.pca(df, numberOfComponents, relevantColumns, columnDescriptions)

def pcaPlot(df, timestamps=None):
    return analysis.pcaPlot(df, timestamps)

def pcaDuoPlot(df_train, df_test_1, df_test_2, plotTitle=None):
    return analysis.pcaDuoPlot(df_train, df_test_1, df_test_2, plotTitle)

def pairplot(df):
    return analysis.pairplot(df)

def scatterplot(df):
    return analysis.scatterplot(df)

def correlationPlot(df, title="Correlation plot"):
    return analysis.correlationPlot(df, title)

def valueDistribution(df, traintime, testtime):
    return analysis.valueDistribution(df, traintime, testtime)

def printCorrelationMatrix(covmat, df, columnNames=None):
    return prints.printCorrelationMatrix(covmat, df, columnNames)

def printExplainedVarianceRatio(pca):
    return prints.printExplainedVarianceRatio(pca)