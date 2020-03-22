def MLP_Dropout(
            self,
            name,
            layers=[128],
            dropout=default_MLP_args['dropout'],
            activation=default_MLP_args['activation'],
            loss=default_MLP_args['loss'],
            optimizer=default_MLP_args['optimizer'],
            metrics=default_MLP_args['metrics'],
            epochs=default_MLP_args['epochs'],
            batchSize=default_MLP_args['batchSize'],
            verbose=default_MLP_args['verbose'],
            validationSize=default_MLP_args['validationSize'],
            testSize=default_MLP_args['testSize']
        ):
        """
        FUNCTION:
            Used to create a Neural Network model using multilayer perceptron
            and reguarlization by dropout
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            layers: list of integers
                List of neuron size for each layer
            dropout: float
                Level of dropout
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        mlpLayers = []
        for layerSize in layers:
            mlpLayers.append([layerSize, activation])

        model = models.kerasSequentialRegressionModelWithDropout(
            params={
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_MLP_args['callbacks'],
                    'enrolWindow': 0,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            structure=mlpLayers,
            dropout=dropout,
        )
        
        return model

    def MLP_Regularized(
            self,
            name,
            layers=[128],
            l1_rate=0.01,
            l2_rate=0.01,
            activation=default_MLP_args['activation'],
            loss=default_MLP_args['loss'],
            optimizer=default_MLP_args['optimizer'],
            metrics=default_MLP_args['metrics'],
            epochs=default_MLP_args['epochs'],
            batchSize=default_MLP_args['batchSize'],
            verbose=default_MLP_args['verbose'],
            validationSize=default_MLP_args['validationSize'],
            testSize=default_MLP_args['testSize']
        ):
        """
        FUNCTION:
            Used to create a Neural Network model using multilayer perceptron
            and reguarlization by Ridge and Lasso regluarization
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            layers: list of integers
                List of neuron size for each layer
            l1_rate: float
                Level of L1 regularization
            l2_rate: float
                Level of L2 regularization
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        mlpLayers = []
        for layerSize in layers:
            mlpLayers.append([layerSize, activation])

        model = models.kerasSequentialRegressionModelWithRegularization(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_MLP_args['callbacks'],
                    'enrolWindow': 0,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            structure = mlpLayers,
            l1_rate=l1_rate,
            l2_rate=l2_rate,
        )
        
        return model

def LSTM_Recurrent(
        self,
        name,
        units=[128],
        dropout=default_LSTM_args['dropout'],
        recurrentDropout=default_LSTM_args['recurrentDropout'],
        training=default_LSTM_args['training'],
        alpha=default_LSTM_args['alpha'],
        activation=default_LSTM_args['activation'],
        loss=default_LSTM_args['loss'],
        optimizer=default_LSTM_args['optimizer'],
        metrics=default_LSTM_args['metrics'],
        epochs=default_LSTM_args['epochs'],
        batchSize=default_LSTM_args['batchSize'],
        verbose=default_LSTM_args['verbose'],
        enrolWindow=default_LSTM_args['enrolWindow'],
        validationSize=default_LSTM_args['validationSize'],
        testSize=default_LSTM_args['testSize'],
        ):
        """
        FUNCTION:
            Used to create a Recurrent Neural Network model using
            Long-Short Term Memory neurons (LSTM). Uses both
            traditional dropout and recurrent dropout for regularization,
            hence the subname _Recurrent
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            units: list of integers
                List of neuron size for each layer
            dropout: float
                Level of dropout
            recurrentDropout: float
                Level of recurrent dropout
            alpha: float
                Alpha of the leaky relu function
            training: boolean
                If the model should apply dropout during prediction or not
                If not, the prediction will be the mean of some number of
                predictions made by the model internally
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.kerasLSTM_Recurrent(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_LSTM_args['callbacks'],
                    'enrolWindow': enrolWindow,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            units=units,
            dropout=dropout,
            recurrentDropout=recurrentDropout,
            training=training,
            alpha=alpha,
        )
        
        return model

def GRU_Recurrent(
        self,
        name,
        units=[128],
        dropout=default_LSTM_args['dropout'],
        recurrentDropout=default_LSTM_args['recurrentDropout'],
        training=default_LSTM_args['training'],
        alpha=default_LSTM_args['alpha'],
        activation=default_LSTM_args['activation'],
        loss=default_LSTM_args['loss'],
        optimizer=default_LSTM_args['optimizer'],
        metrics=default_LSTM_args['metrics'],
        epochs=default_LSTM_args['epochs'],
        batchSize=default_LSTM_args['batchSize'],
        verbose=default_LSTM_args['verbose'],
        enrolWindow=default_LSTM_args['enrolWindow'],
        validationSize=default_LSTM_args['validationSize'],
        testSize=default_LSTM_args['testSize'],
        ):
        """
        FUNCTION:
            Used to create a Recurrent Neural Network model using
            Long-Short Term Memory neurons (LSTM). Uses both
            traditional dropout and recurrent dropout for regularization,
            hence the subname _Recurrent
        
        PARAMS:
            name: str
                A name/alias given to the model by the user
            units: list of integers
                List of neuron size for each layer
            dropout: float
                Level of dropout
            recurrentDropout: float
                Level of recurrent dropout
            alpha: float
                Alpha of the leaky relu function
        
        RETURNS:
            model: MachineLearningModel
                Object with typical machine learning methods like train, predict etc.
        """

        model = models.kerasLSTM_Recurrent(
            params = {
                'name': name,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'args': {
                    'activation': activation,
                    'loss': loss,
                    'optimizer': optimizer,
                    'metrics': metrics,
                    'epochs': epochs,
                    'batchSize': batchSize,
                    'verbose': verbose,
                    'callbacks': default_LSTM_args['callbacks'],
                    'enrolWindow': enrolWindow,
                    'validationSize': validationSize,
                    'testSize': testSize,
                },
            },
            units=units,
            dropout=dropout,
            recurrentDropout=recurrentDropout,
            training=training,
            alpha=alpha,
        )
        
        return model