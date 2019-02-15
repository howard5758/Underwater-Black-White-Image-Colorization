'''Colorization autoencoder

The autoencoder is trained with grayscale images as input
and colored images as output.
Colorization autoencoder can be treated like the opposite
of denoising autoencoder. Instead of removing noise, colorization
adds noise (color) to the grayscale image.

Grayscale Images --> Colorization --> Color Images
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use("Agg")

import argparse
import cv2
import sys

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


# convert from color image (RGB) to grayscale
# source: opencv.org
# grayscale = 0.299*red + 0.587*green + 0.114*blue
def rgb2gray(rgb):
    print(rgb.shape)
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# loop through the directory on the command line to
# read in the training and test images
# we use the cv2 generatlized reader

dir = sys.argv[1]

imgs_in = []

for root, _, files in os.walk(dir):
    for f in files:
        fullpath = os.path.join(root, f)

        img = cv2.imread(fullpath)

        imgs_in.append(img)

imgs_array = np.array(imgs_in)        


# call scikit-learn function to partition into random train and test sets
# we call with the img_arry twice since we dont have labels and need an Y with same length
# recall that _ are throw aways
x_train, x_test, _, _ = train_test_split(imgs_array, imgs_array, test_size= 0.20, random_state=42)

# input image dimensions
# we assume data format "channels_last"
print(x_train.shape)

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

# create saved_images folder
imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(), imgs_dir)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

# display the 1st 25 input images (color and gray)
imgs = x_test[:25]
imgs = imgs.reshape((5, 5, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

# convert color train and test images to gray
# we are using a custom formula that leaves the images in 3 channels
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

print(x_train_gray.shape)

# display grayscale version of test images
imgs = x_test_gray[:25]
imgs = imgs.reshape((5, 5, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()

########################################### 
# test segmentation
temp = x_train_gray[0]
print("temp shape:")
print(temp.shape)
segments_fz = felzenszwalb(temp, scale=100, sigma=0.5, min_size=3)
print(segments_fz)

plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
temp = mark_boundaries(temp, segments_fz)
plt.imshow(temp/255, interpolation='none')
plt.savefig('%s/test_seg.png' % imgs_dir)
#plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

###########################################
# test seg color
gray = mark_boundaries(x_train_gray[0], segments_fz)
target = x_test[0]
seg = segments_fz
seg_mean = {}
print(seg[0][0])
print(target[0, 0, :])
print(target[0, 0, 1])
for i in range(0, img_rows):
  for j in range(0, img_cols):
    if not seg[i][j] in seg_mean:
      seg_mean[seg[i][j]] = [int(target[i, j, 0]), int(target[i, j, 1]), int(target[i, j, 2]), 1]
    else:
      cur = seg_mean[seg[i][j]]
      cur[0] += int(target[i, j, 0])
      cur[1] += int(target[i, j, 1])
      cur[2] += int(target[i, j, 2])
      cur[3] += 1
print(seg_mean[0])
for segg in seg_mean:
  seg_mean[segg][0] = seg_mean[segg][0]/seg_mean[segg][3]
  seg_mean[segg][1] = seg_mean[segg][1]/seg_mean[segg][3]
  seg_mean[segg][2] = seg_mean[segg][2]/seg_mean[segg][3]
print(seg_mean[0])
print(gray.shape)

for i in range(0, img_rows):
  for j in range(0, img_cols):
    gray[i, j, 0] = seg_mean[seg[i][j]][0]
    gray[i, j, 1] = seg_mean[seg[i][j]][1]
    gray[i, j, 2] = seg_mean[seg[i][j]][2]


plt.figure()
plt.axis('off')
plt.title('Test seg color')
temp = mark_boundaries(temp, segments_fz)
plt.imshow(gray/255, interpolation='none')
plt.savefig('%s/test_seg_color.png' % imgs_dir)
#plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

plt.figure()
plt.axis('off')
plt.title('First color')
plt.imshow(target, interpolation='none')
plt.savefig('%s/first_color.png' % imgs_dir)
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(temp, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(temp, segments_fz))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(temp, segments_fz))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(temp, segments_fz))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()


# normalize output train and test color images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# normalize input train and test grayscale images
x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255

# reshape images to row x col x channel for CNN output/validation
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

# reshape images to row x col x channel for CNN input
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)
all_x = np.concatenate((x_train_gray, x_test_gray), axis=0)

# network parameters
input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256
# encoder/decoder number of CNN layers and filters per layer
layer_filters = [64, 128, 256]

# build the autoencoder model
# first build the encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
# shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
shape = K.int_shape(x)

# generate a latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# instantiate encoder model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# build the decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

outputs = Conv2DTranspose(filters=channels,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

# save weights for future use (e.g. reload parameters w/o training)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')

# called every epoch
callbacks = [lr_reducer, checkpoint]

# train the autoencoder
autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=2,
                batch_size=batch_size,
                callbacks=callbacks)

# predict the autoencoder output from test data
x_decoded = autoencoder.predict(x_test_gray)
print(x_decoded.shape)
x_decoded = x_decoded
# display the 1st 100 colorized images
imgs = x_decoded[:25]
imgs = imgs.reshape((5, 5, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png' % imgs_dir)
plt.show()

############################################
# actual seg color on predicted images
x_decoded = x_decoded * 255
for idx in range(0, 25):
  print(idx)
  target = x_decoded[idx]
  print(target.shape)
  temp = x_test_gray[idx]
  print(temp.shape)
  segments_fz = felzenszwalb(temp, scale=100, sigma=0.5, min_size=3)

  ###########################################
  # test seg color
  # temp = mark_boundaries(temp, segments_fz)
  # just for testing
  temp = x_decoded[idx + 25]
  seg = segments_fz
  seg_mean = {}

  for i in range(0, img_rows):
    for j in range(0, img_cols):
      if not seg[i][j] in seg_mean:
        seg_mean[seg[i][j]] = [int(target[i, j, 0]), int(target[i, j, 1]), int(target[i, j, 2]), 1]
      else:
        cur = seg_mean[seg[i][j]]
        cur[0] += int(target[i, j, 0])
        cur[1] += int(target[i, j, 1])
        cur[2] += int(target[i, j, 2])
        cur[3] += 1
  # print(seg_mean)
  for segg in seg_mean:
    seg_mean[segg][0] = seg_mean[segg][0]/seg_mean[segg][3]
    seg_mean[segg][1] = seg_mean[segg][1]/seg_mean[segg][3]
    seg_mean[segg][2] = seg_mean[segg][2]/seg_mean[segg][3]
  # print(seg_mean)

  for i in range(0, img_rows):
    for j in range(0, img_cols):
      temp[i, j, 0] = seg_mean[seg[i][j]][0]
      temp[i, j, 1] = seg_mean[seg[i][j]][1]
      temp[i, j, 2] = seg_mean[seg[i][j]][2]

  x_decoded[idx] = temp
# print(target)
# print(x_decoded[0].shape)
# print(x_decoded[0])
imgs = x_decoded[:25]
imgs = imgs.reshape((5, 5, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Actual Seg Color (Predicted)')
plt.imshow(imgs/255, interpolation='none')
plt.savefig('%s/seg_colorized.png' % imgs_dir)
plt.show()

chrome = x_decoded[:25]
lum = x_test_gray[:25]*255

# print(chrome.shape)
# print(chrome[0])
# print(lum.shape)
# print(lum[0])

for i in range(0, 25):
  for r in range(0, img_rows):
    for j in range(0, img_cols):
      gray_mean = chrome[i, r, j, 0] * 0.299 + chrome[i, r, j, 1] * 0.587 + chrome[i, r, j, 2] * 0.114
      diff = lum[i, r, j]/gray_mean
      chrome[i, r, j, 0] = max(min(255, chrome[i, r, j, 0] * diff), 0)
      chrome[i, r, j, 1] = max(min(255, chrome[i, r, j, 1] * diff), 0)
      chrome[i, r, j, 2] = max(min(255, chrome[i, r, j, 2] * diff), 0)

imgs = chrome
imgs = imgs.reshape((5, 5, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('chrom+lum (Predicted)')
plt.imshow(imgs/255, interpolation='none')
plt.savefig('%s/chrome+lum_colorized.png' % imgs_dir)
plt.show()




##################################################################################################################################
# # train from decoded
# new_input = autoencoder.predict(all_x)
# new_x_train = new_input[:1124]
# new_x_test = new_input[1124:]


# # network parameters
# input_shape = (img_rows, img_cols, 3)
# batch_size = 32
# kernel_size = 3
# latent_dim = 256
# # encoder/decoder number of CNN layers and filters per layer
# layer_filters = [64, 128, 256]

# # build the autoencoder model
# # first build the encoder model
# inputs = Input(shape=input_shape, name='encoder_input')
# x = inputs
# # stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
# for filters in layer_filters:
#     x = Conv2D(filters=filters,
#                kernel_size=kernel_size,
#                strides=2,
#                activation='relu',
#                padding='same')(x)

# # shape info needed to build decoder model so we don't do hand computation
# # the input to the decoder's first Conv2DTranspose will have this shape
# # shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
# shape = K.int_shape(x)

# # generate a latent vector
# x = Flatten()(x)
# latent = Dense(latent_dim, name='latent_vector')(x)

# # instantiate encoder model
# encoder = Model(inputs, latent, name='encoder')
# encoder.summary()

# # build the decoder model
# latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
# x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
# x = Reshape((shape[1], shape[2], shape[3]))(x)

# # stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
# for filters in layer_filters[::-1]:
#     x = Conv2DTranspose(filters=filters,
#                         kernel_size=kernel_size,
#                         strides=2,
#                         activation='relu',
#                         padding='same')(x)

# outputs = Conv2DTranspose(filters=channels,
#                           kernel_size=kernel_size,
#                           activation='sigmoid',
#                           padding='same',
#                           name='decoder_output')(x)

# # instantiate decoder model
# decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()

# # autoencoder = encoder + decoder
# # instantiate autoencoder model
# autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
# autoencoder.summary()

# # reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                verbose=1,
#                                min_lr=0.5e-6)

# # Mean Square Error (MSE) loss function, Adam optimizer
# autoencoder.compile(loss='mse', optimizer='adam')

# # called every epoch
# callbacks = [lr_reducer]

# # train the autoencoder
# autoencoder.fit(new_x_train,
#                 x_train,
#                 validation_data=(new_x_test, x_test),
#                 epochs=100,
#                 batch_size=batch_size,
#                 callbacks=callbacks)

# # predict the autoencoder output from test data
# x_decoded = autoencoder.predict(new_x_test)
# print(x_decoded.shape)
# # x_decoded = x_decoded * 255
# # display the 1st 100 colorized images
# imgs = x_decoded[:25]
# imgs = imgs.reshape((5, 5, img_rows, img_cols, channels))
# imgs = np.vstack([np.hstack(i) for i in imgs])
# plt.figure()
# plt.axis('off')
# plt.title('Reinforcement Colorized test images (Predicted)')
# plt.imshow(imgs, interpolation='none')
# plt.savefig('%s/reinforcement_colorized.png' % imgs_dir)
# plt.show()