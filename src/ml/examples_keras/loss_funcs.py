"""
calling:
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
or
model.compile(loss='mean_squared_error', optimizer='sgd')

---------------------------------------------------------

mean_squared_error
keras.losses.mean_squared_error(y_true, y_pred)

mean_absolute_error
keras.losses.mean_absolute_error(y_true, y_pred)

mean_absolute_percentage_error
keras.losses.mean_absolute_percentage_error(y_true, y_pred)

mean_squared_logarithmic_error
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)

squared_hinge
keras.losses.squared_hinge(y_true, y_pred)

hinge
keras.losses.hinge(y_true, y_pred)

categorical_hinge
keras.losses.categorical_hinge(y_true, y_pred)

logcosh
keras.losses.logcosh(y_true, y_pred)

huber_loss
keras.losses.huber_loss(y_true, y_pred, delta=1.0)

categorical_crossentropy
keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)

sparse_categorical_crossentropy
keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)

binary_crossentropy
keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)

kullback_leibler_divergence
keras.losses.kullback_leibler_divergence(y_true, y_pred)

poisson
keras.losses.poisson(y_true, y_pred)

cosine_proximity
keras.losses.cosine_proximity(y_true, y_pred, axis=-1)

is_categorical_crossentropy
keras.losses.is_categorical_crossentropy(loss)

"""