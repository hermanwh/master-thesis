from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error

def calculateR2Score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

def calculateMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

def calculateMAE(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def calculateMaxError(y_true, y_pred):
    if len(y_true.shape) > 1 and y_true.shape and y_true.shape[1] > 1:
        maxerror = None
    else:
        maxerror = max_error(y_true, y_pred)
    return maxerror

def calculateMetrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    if len(y_true.shape) > 1 and y_true.shape and y_true.shape[1] > 1:
        maxerror = None
    else:
        maxerror = max_error(y_true, y_pred)
    return [r2, mse, mae, maxerror]