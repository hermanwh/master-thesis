
1. Determine input shape, e.g. size 10

-----------------------------------------------------

2. Build model using e.g. sequential, dense, activation etc.

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(INPUT_SHAPE,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

---------------------------------------------------

3. Compile model
https://keras.io/models/sequential/

params:
- optimizer, 
- loss=None, 
- metrics=None, 
- loss_weights=None, 
- sample_weight_mode=None, 
- weighted_metrics=None
- target_tensors=None

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

-----------------------------------------------------

4. Training
https://keras.io/models/sequential/

params:
- x=None
- y=None
- batch_size=None
- epochs=1
- verbose=1
- callbacks=None
- validation_split=0.0
- validation_data=None
- shuffle=True
- class_weight=None
- sample_weight=None
- initial_epoch=0
- steps_per_epoch=None
- validation_steps=None
- validation_freq=1
- max_queue_size=10
- workers=1
- use_multiprocessing=False

model.fit(data, labels, epochs=10, batch_size=32)

-----------------------------------------------------

5. Evaluation
params:
x=None
y=None
batch_size=None
verbose=1
sample_weight=None
steps=None
callbacks=None
max_queue_size=10
workers=1
use_multiprocessing=False

-----------------------------------------------------

6. Predict
params:
x
batch_size=None
verbose=0
steps=None
callbacks=None
max_queue_size=10
workers=1
use_multiprocessing=False

