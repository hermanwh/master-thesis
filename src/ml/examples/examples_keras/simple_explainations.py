
Sample: one element of a dataset, e.g. one image, audio file or dataset row

Batch: set of N samples. All samples in a batch are processed independently, in parallell
    - one batch results in one update to the model

Epoch: arbitrary cutoff, "one pass over the entire dataset"
    - used to separate training into distinct phases, for logging/periodic evaluations

Callbacks: runs at the end of an epoch, e.g. learning rate changes or model checkpoints (saving)







# Saving:

saved architecture, weights, training config and optimizer state

from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')

------------------------

ONLY WEIGHTS:
model.save_weights('my_model_weights.h5')


# Loading:

from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)

---------------------
ONLY WEIGHTS:
model.load_weights('my_model_weights.h5')

Same weights, different architecture:
model.load_weights('my_model_weights.h5', by_name=True)




# Training on data too large for memory:

You can do batch training using model.train_on_batch(x, y)
and model.test_on_batch(x, y)



#Pre-trained models:

Xception
VGG16
VGG19
ResNet
ResNet v2
ResNeXt
Inception v3
Inception-ResNet v2
MobileNet v1
MobileNet v2
DenseNet
NASNet

from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnext import ResNeXt50
from keras.applications.resnext import ResNeXt101
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile

model = VGG16(weights='imagenet', include_top=True)
