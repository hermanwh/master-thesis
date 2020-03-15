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