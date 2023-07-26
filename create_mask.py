'''
Generate layer specific images for PredNet on KITTI sequences.
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input
from data_utils_seg import SequenceGenerator
from predirep_seg import PredNet  # load network
from PIL import Image
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deactivate GPU usage

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# Parameters
nt = 10  # number time steps
data_dir = '/home/devel/Projects/rl_dataset/dataset-hbp/img/4'
out_dir = 'test_out'

# Load images
imgs = os.listdir(data_dir)
imgs.sort(key=natural_keys)
imgs = [ Image.open(os.path.join(data_dir, i)) for i in imgs ]
img_shape = ( imgs[0].width, imgs[0].height)

# Load trained model
weights_file = "weights/predirep_weights.hdf5"
json_file = "weights/predirep_model.json"

f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
input_shape[1] = input_shape[1]  # height - can be increased/decreased but needs to be divisible by 8
input_shape[2] = input_shape[2]  # width - can be increased/decreased but needs to be divisible by 8
inputs = Input(shape=tuple(input_shape))

# Convert images to numpy array
np_imgs = [ i.resize((input_shape[2], input_shape[1])) for i in imgs ]
np_imgs = np.stack(np_imgs, axis=0)
np_imgs = np_imgs[np.newaxis, 0:nt, ...]
np_imgs = np_imgs.astype(np.float32) / 255.0
X_test = np_imgs

layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'segmentation'

# Create mask - for your own data create a different X_test. Has to have shape (batch_size, nt, height, width, color)
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)
mask = test_model.predict(X_test)  # output shape will be (batch_size, nt, height, width, color). Channels are r -> pringles, g -> mustard, b-> background

# Save images in out_dir
os.makedirs(out_dir, exist_ok=True)
mask_imgs = [ Image.fromarray((mask[0, i, ...]*255).astype(np.uint8)).resize(img_shape, resample=Image.NEAREST) for i in range(0, mask.shape[1]) ]
for i in range(0, len(mask_imgs)):
    imgs[i].save(os.path.join(out_dir, f'img_{i}.png'))
    mask_imgs[i].save(os.path.join(out_dir, f'mask_{i}.png'))

