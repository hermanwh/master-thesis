import sys, os
ROOT_PATH = os.path.abspath(".").split("src")[0]
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/src/utils/"))
if module_path not in sys.path:
    sys.path.append(module_path)

from keras.models import load_model
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

import plots
import prints
import pickle
import numpy as np

np.random.seed(100)

def printModelSummary(model):
    if hasattr(model, "summary"):
        # Keras Model object
        print(model.summary())
    elif hasattr(model, "model"):
        # MachineLearningModel object
        if hasattr(model.model, "summary"):
            # MachineLearningModel.model will be a Keras Model object
            printModelSummary(model.model)
    elif hasattr(model, "models"):
        # EnsembleModel object
        print("Model is of type Ensemble Model")
        print("Sub model summaries will follow")
        print("-------------------------------")
        for mod in model.models:
            # EnsembleModel.models will be a list of MachineLearningModels
            printModelSummary(mod)
    else:
        print("Simple models have no summary")
    
def printModelWeights(model):
    if hasattr(model, "summary"):
        # Keras Model object
        for layer in model.layers: print(layer.get_config(), layer.get_weights())
    elif hasattr(model, "model"):
        # MachineLearningModel object
        if hasattr(model.model, "summary"):
            # MachineLearningModel.model will be a Keras Model object
            printModelSummary(model.model)
    elif hasattr(model, "models"):
        # EnsembleModel object
        print("Model is of type Ensemble Model")
        print("Sub model summaries will follow")
        print("-------------------------------")
        for mod in model.models:
            # EnsembleModel.models will be a list of MachineLearningModels
            printModelSummary(mod)
    else:
        if hasattr(model, "get_params"):
            print(model.get_params())
        else:
            print("No weights found")

def plotKerasModel(model):
    plot_model(model.model)

def getBasicCallbacks(monitor="val_loss", patience_es=200, patience_rlr=80):
    return [
        EarlyStopping(
            monitor = monitor, min_delta = 0.00001, patience = patience_es, mode = 'auto', restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = monitor, factor = 0.5, patience = patience_rlr, verbose = 1, min_lr=5e-4,
        )
    ]

def getBasicHyperparams():
    return {
        'activation': 'relu',
        'loss': 'mean_squared_error',
        'optimizer': 'adam',
        'metrics': ['mean_squared_error'],
    }

def trainModels(modelList, filename, targetColumns, retrain=False, save=True):
    if retrain:
        for mod in modelList:
            print("Training model " + mod.name)
            mod.train()
    else:
        for mod in modelList:
            if mod.modelType != "Ensemble":
                loadedModel, loadedHistory = loadModel(mod.name, filename, targetColumns)
                if loadedModel is not None:
                    print("Model " + mod.name + " was loaded from file")
                    mod.model = loadedModel
                    mod.history = loadedHistory
                else:
                    print("Training model " + mod.name)
                    mod.train()
            else:
                for model in mod.models:
                    loadedModel, loadedHistory = loadModel(model.name, filename, targetColumns, ensembleName=mod.name)
                    if loadedModel is not None:
                        print("Model " + mod.name + " was loaded from file")
                        model.model = loadedModel
                        model.history = loadedHistory
                    else:
                        print("Training submodel " + model.name + " of Ensemble " + mod.name)
                        model.train()

                mod.trainEnsemble()

    if save:
        saveModels(modelList, filename, targetColumns)

    trainingSummary = getTrainingSummary(modelList)
    prints.printTrainingSummary(trainingSummary)
    plots.plotTrainingSummary(trainingSummary)
              
def loadModel(modelname, filename, targetColumns, ensembleName=None):
    subdir = filename.split('/')[-2]
    datafile = filename.split('/')[-1].split('.')[0]
    joinedColumns = "_".join(targetColumns)
    
    modName = "_".join(modelname.split(' '))
    if ensembleName is None:
        directory = ROOT_PATH + '/src/ml/trained_models/' + subdir + '/' + datafile + '/' + modName + '_' + joinedColumns
    else:    
        ensName = "_".join(ensembleName.split(' '))
        directory = ROOT_PATH + '/src/ml/trained_models/' + subdir + '/' + datafile + '/' + ensName + '_' + joinedColumns + '/' + modName
    if os.path.isfile((directory + ".h5")) and os.path.isfile((directory + ".h5")):
        model = load_model(directory + ".h5")
        history = pickle.load(open(directory + ".pickle", "rb"))
    else:
        model = None
        history = None
    return [model, history]

def saveModels(modelList, filename, targetColumns):
    subdir = filename.split('/')[-2]
    datafile = filename.split('/')[-1].split('.')[0]
    joinedColumns = "_".join(targetColumns)
    
    for model in modelList:
        modName = "_".join(model.name.split(' '))
        directory = ROOT_PATH + '/src/ml/trained_models/' + subdir + '/' + datafile + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        modelPath = directory
        modelName = modName + '_' + joinedColumns
        metricsPath = directory + modName + '_' + joinedColumns + ".txt"
        model.save(modelPath, modelName)

def getTrainingSummary(modelList):
    loss_dict = {}
    for model in modelList:
        if model.modelType != "Ensemble":
            if model.history is not None:
                loss = model.history['loss']
                val_loss = model.history['val_loss']
                loss_best = np.amin(loss)
                loss_loc = np.where(loss == loss_best)[0]
                val_loss_best = np.amin(val_loss)
                val_loc = np.where(val_loss == val_loss_best)[0]
                loss_dict[model.name] = {
                    'loss': loss,
                    'val_loss': val_loss,
                    'loss_final': loss_best,
                    'loss_loc': loss_loc,
                    'val_loss_final': val_loss_best,
                    'val_loss_loc': val_loc,
                    'length': len(loss),
                }
        else:
            for submodel in model.models:
                if submodel.history is not None:
                    loss = submodel.history['loss']
                    val_loss = submodel.history['val_loss']
                    loss_dict[model.name + " " + submodel.name] = {
                        'loss': loss,
                        'val_loss': val_loss,
                        'loss_final': loss[-1],
                        'val_loss_final': val_loss[-1],
                        'length': len(loss)
                    }
    return loss_dict