calling:
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
or 
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])

------------------------------------

accuracy
keras.metrics.accuracy(y_true, y_pred)

binary_accuracy
keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)

categorical_accuracy
keras.metrics.categorical_accuracy(y_true, y_pred)

sparse_categorical_accuracy
keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

top_k_categorical_accuracy
keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

sparse_top_k_categorical_accuracy
keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)

cosine_proximity
keras.metrics.cosine_proximity(y_true, y_pred, axis=-1)

clone_metric
keras.metrics.clone_metric(metric)

----------------------------

Custom metrics:

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
