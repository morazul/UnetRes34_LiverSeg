# Build ResNet34 + Unet Model
import numpy as np
# import pandas as pd
# import six
from utils import *
# from random import randint

import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# import seaborn as sns
# sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

import keras
import keras.utils
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
# from keras.utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

BATCH_SIZE = 32
EDGE_CROP = 16
NB_EPOCHS = 5
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 100
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False

im_width = 512
im_height = 512
classes = 3 # liver, lesion, background
path_train = '/Users/Mor\'s Yoga/Documents/ComputerVision/LiverProject/Data'

# Load data
X, y = get_data(path_train, im_height, im_width, classes, train=True)
X, y = X[:320], y[:320]

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=2019)


# seg_model = UResNet34(input_shape=(768,768,3),encoder_weights=True)
seg_model = UResNet34(input_shape=(im_height, im_width, 3), classes=classes, encoder_weights=True)
seg_model.summary()

# Set training check point
weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

# Compile model
# seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=bce_logdice_loss,
#                   metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=categorical_crossentropy,
                  metrics=[dice_coef, 'categorical_accuracy', true_positive_rate])


# early_stopping = EarlyStopping(patience=10, verbose=1)
# model_checkpoint = ModelCheckpoint(weight_path, save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.00001, verbose=1)

# Training
epochs = 1
batch_size = 32

step_count = min(MAX_TRAIN_STEPS, X_train.shape[0]//BATCH_SIZE)
# valid_step_count = min(MAX_TRAIN_STEPS, X_valid.shape[0]//BATCH_SIZE)
valid_step_count = X_valid.shape[0]
# aug_gen = create_aug_gen(make_image_gen(balanced_train_df))
# loss_history = [seg_model.fit_generator(aug_gen,
#                             steps_per_epoch=step_count,
#                             validation_data=(valid_x, valid_y),
#                             epochs=epochs,
#                             callbacks=callbacks_list,shuffle=True)]
loss_history = [seg_model.fit(X_train,
                              y_train,
                              batch_size=batch_size,
                              # steps_per_epoch=step_count,
                              validation_data=(X_valid, y_valid),
                              epochs=epochs,
                              # validation_steps=valid_step_count,
                              callbacks=callbacks_list, shuffle=True)]

seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')

# Plot loss
epich = np.cumsum(np.concatenate([np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
_ = ax1.plot(epich, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
             epich, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax2.plot(epich, np.concatenate([mh.history['dice_coef'] for mh in loss_history]), 'b-',
             epich, np.concatenate([mh.history['val_dice_coef'] for mh in loss_history]), 'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('DICE')

# Full model
from keras import models, layers
if IMG_SCALING is not None:
    fullres_model = models.Sequential()
    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
    fullres_model.add(seg_model)
    fullres_model.add(layers.UpSampling2D(IMG_SCALING))
else:
    fullres_model = seg_model
fullres_model.save('fullres_model_net34.h5')

# # Test
from skimage.io import imread
test_image_dir = "/Users/Mor\'s Yoga/Documents/ComputerVision/LiverProject/temp"
test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')
fig, m_axs = plt.subplots(20, 2, figsize = (10, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    first_img = np.expand_dims(c_img, 0)/255.0
    # first_seg = fullres_model.predict(first_img)
    first_seg = seg_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0,:, :, 0])
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')