def kerasSequentialRegressionModelWithRegularization(
    params,
    structure,
    l1_rate=0.01,
    l2_rate=0.01,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])

    model = Sequential()

    firstLayerNeurons, firstLayerActivation = structure[0]
    model.add(
        Dense(
            firstLayerNeurons,
            input_dim=X_train.shape[1],
            activation=firstLayerActivation,
            kernel_regularizer=l2(l2_rate),
            activity_regularizer=l1(l1_rate),
        )
    )
    
    for neurons, activation in structure[1:]:
        model.add(
            Dense(
                neurons,
                activation=activation,
                kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l1_rate),
            )
        )
    
    model.add(
        Dense(
            y_train.shape[1],
            activation='linear',
        )
    )

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="MLP",
        name=name,
    )

def kerasLSTM(params,
    units=[128],
    dropout=None,
    alpha=None,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])

    model = Sequential()

    if len(units) > 1:
        firstLayerUnits = units[0]
        model.add(
            LSTM(
                firstLayerUnits,
                activation = args.activation,
                return_sequences=True,
                input_shape=(args.enrolWindow, X_train.shape[1]),
            )
        )
        if alpha is not None:
            model.add(LeakyReLU(alpha=alpha))
        if dropout is not None:
            model.add(Dropout(dropout))

        for layerUnits in units[1:]:
            model.add(
                LSTM(
                    layerUnits,
                    activation = args.activation,
                    return_sequences=False
                )
            )
            if alpha is not None:
                model.add(LeakyReLU(alpha=alpha))
            if dropout is not None:
                model.add(Dropout(dropout))
    else:
        firstLayerUnits = units[0]
        model.add(LSTM(firstLayerUnits, input_shape=(args.enrolWindow, X_train.shape[1])))
        if alpha is not None:
            model.add(LeakyReLU(alpha=alpha))
        if dropout is not None:
            model.add(Dropout(dropout))

    model.add(
        Dense(
            y_train.shape[1],
            activation='linear',
        )
    )

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="RNN",
        name=name,
    )

def kerasGRU(
    params,
    units=[128],
    dropout=None,
    alpha=None,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])

    model = Sequential()

    if len(units) > 1:
        firstLayerUnits = units[0]
        model.add(
            GRU(
                firstLayerUnits,
                return_sequences=True,
                input_shape=(args.enrolWindow, X_train.shape[1]),
            )
        )
        if alpha is not None:
            model.add(LeakyReLU(alpha=alpha))
        if dropout is not None:
            model.add(Dropout(dropout))

        for layerUnits in units[1:]:
            model.add(
                GRU(
                    layerUnits,
                    return_sequences=False,
                )
            )
            if alpha is not None:
                model.add(LeakyReLU(alpha=alpha))
            if dropout is not None:
                model.add(Dropout(dropout))
    else:
        firstLayerUnits = units[0]
        model.add(
            GRU(
                firstLayerUnits,
                input_shape=(args.enrolWindow, X_train.shape[1]),
            )
        )
        if alpha is not None:
            model.add(LeakyReLU(alpha=alpha))
        if dropout is not None:
            model.add(Dropout(dropout))

    model.add(
        Dense(
            y_train.shape[1],
            activation='linear',
        )
    )

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="RNN",
        name=name,
    )

def kerasSequentialRegressionModel(
    params,
    structure,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])

    model = Sequential()

    firstLayerNeurons, firstLayerActivation = structure[0]
    model.add(
        Dense(
            firstLayerNeurons,
            input_dim=X_train.shape[1],
            activation=firstLayerActivation
        )
    )
    
    for neurons, activation in structure[1:]:
        model.add(
            Dense(
                neurons,
                activation=activation,
            )
        )
    
    model.add(
        Dense(
            y_train.shape[1],
            activation='linear',
        )
    )

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="MLP",
        name=name,
    )

def kerasSequentialRegressionModelWithDropout(
    params,
    structure,
    dropout=0.2,
    l1_rate=0.01,
    l2_rate=0.01,
    ):

    X_train = params['X_train']
    y_train = params['y_train']
    name = params['name']
    args = Args(params['args'])

    model = Sequential()

    firstLayerNeurons, firstLayerActivation = structure[0]
    model.add(
        Dense(
            firstLayerNeurons,
            input_dim=X_train.shape[1],
            activation=firstLayerActivation,
            kernel_regularizer=l2(l2_rate),
            activity_regularizer=l1(l1_rate),
        )
    )
    model.add(Dropout(dropout))

    for neurons, activation in structure[1:]:
        model.add(
            Dense(
                neurons,
                activation=activation,
            )
        )
        model.add(Dropout(dropout))
    
    model.add(
        Dense(
            y_train.shape[1],
            activation='linear',
        )
    )

    return MachinLearningModel(
        model,
        X_train,
        y_train,
        args=args,
        modelType="MLP",
        name=name,
    )