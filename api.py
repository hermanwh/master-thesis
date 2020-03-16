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

from src.ml.analysis.covmat import (covmat, printCovMat)
from src.ml.analysis.pca import (pca, printExplainedVarianceRatio)
from src.ml.analysis.pcaPlot import (pcaPlot, printReconstructionRow)

class Api():

    def __init__(self):
        self.filename = None
        self.relevantColumns = None
        self.columnDescriptions = None
        self.df = None
        self.traintime = None
        self.testtime = None
        self.df_train = None
        self.df_test = None
        self.targetColumns = None
        self.modelList = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.maxEnrolWindow = None
        self.indexColumn = None
    
    def initDataframe(self, filename, relevantColumns, columnDescriptions):
        self.filename = filename
        self.relevantColumns = relevantColumns
        self.columnDescriptions = columnDescriptions
        df = utilities.initDataframe(filename, relevantColumns, columnDescriptions)
        self.df = df
        return df

    def printMaxEnrol(self):
        print(self.maxEnrolWindow)

    def getTestTrainSplit(self, traintime, testtime):
        self.traintime = traintime
        self.testtime = testtime
        df_train, df_test = utilities.getTestTrainSplit(self.df, traintime, testtime)
        self.df_train = df_train
        self.df_test = df_test
        return [df_train, df_test]

    def getFeatureTargetSplit(self, targetColumns):
        self.targetColumns = targetColumns
        X_train, y_train, X_test, y_test =  utilities.getFeatureTargetSplit(self.df_train, self.df_test, targetColumns)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return [X_train, y_train, X_test, y_test]

    def prepareDataframe(self, df, traintime, testtime, targetColumns):
        df_train, df_test = getTestTrainSplit(df, traintime, testtime)
        return getFeatureTargetSplit(df_train, df_test, targetColumns)

    def initModels(self, modelList):
        self.maxEnrolWindow = utilities.findMaxEnrolWindow(modelList)
        self.indexColumn = self.df_test.iloc[self.maxEnrolWindow:].index
        self.modelList = modelList

    def trainModels(self, retrain=False):
        utilities.trainModels(self.modelList, self.filename, self.targetColumns, retrain)

    def predictWithModels(self, plot=True):
        modelNames, metrics_train, metrics_test, deviationsList, columnsList = utilities.predictWithModels(
            self.modelList,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.targetColumns 
        )
        if plot:
            utilities.plotModelPredictions(
                plt,
                deviationsList,
                columnsList,
                self.indexColumn,
                self.columnDescriptions,
                self.traintime
            )
            utilities.printModelPredictions(modelNames, metrics_train, metrics_test)
        return [modelNames, metrics_train, metrics_test]