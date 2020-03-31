from keras import optimizers

def getOptimizerSGD(learning_rate=0.01, momentum=0.0, nesterov=False):
    return optimizers.SGD(learning_rate=learning_rate, momentum=learning_rate, nesterov=nesterov)

def getOptimizerRMSprop(learning_rate=0.0001, rho=0.9):
    return optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

def getOptimizerAdagrad(learning_rate=0.01):
    return optimizers.Adagrad(learning_rate=learning_rate)

def getOptimizerAdadelta(learning_rate=0.0001, rho=0.95):
    return optimizers.Adadelta(learning_rate=learning_rate, rho=rho)

def getOptimizerAdam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False):
    return optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

def getOptimizerAdamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999):
    return optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

def getOptimizerNadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999):
    return optimizers.Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    