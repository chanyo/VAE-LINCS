'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

# Reference

- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import norm

import scipy.io
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10 #mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy.io import loadmat
import pandas as pd
import cPickle
import cv2
import hdf5storage

# input image dimensions
img_rows, img_cols, img_chns = 96, 96, 1 # mnist
#img_rows, img_cols, img_chns = 28, 28, 1 # mnist
# number of convolutional filters to use
filters = 24
# convolution kernel size
num_conv = 4

batch_size = 100 # 100
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 20
intermediate_dim = 128

epsilon_std = 1.0
epochs = 500

x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                #kernel_size=(2, 2),
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
conv_5 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_4)
flat = Flatten()(conv_5)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * img_rows / 2 * img_cols / 2, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, img_rows / 2, img_cols / 2)
else:
    output_shape = (batch_size, img_rows / 2 , img_cols / 2, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, img_rows/2+1, img_cols/2+1)
else:
    output_shape = (batch_size, img_rows/2+1, img_cols/2+1, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
  def __init__(self, **kwargs):
    self.is_placeholder = True
    super(CustomVariationalLayer, self).__init__(**kwargs)

  def vae_loss(self, x, x_decoded_mean_squash):
    x = K.flatten(x)
    x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
    xent_loss = img_rows*img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

  def call(self, inputs):
    x = inputs[0]
    x_decoded_mean_squash = inputs[1]
    loss = self.vae_loss(x, x_decoded_mean_squash)
    self.add_loss (loss, inputs = inputs)
    return x


# instantiate VAE model
y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model (x,y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


# # instantiate VAE model
# vae = Model(x, x_decoded_mean_squash)


# print('x.shape:', x.shape)
print('x_decoded_mean_squash.shape:', x_decoded_mean_squash.shape)


# # Compute VAE loss
# xent_loss = img_chns * img_rows * img_cols * metrics.binary_crossentropy(
#     K.flatten(x),K.flatten(x_decoded_mean_squash))

# print('xent_loss:', xent_loss)


# kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
# vae_loss = K.mean(xent_loss + kl_loss)
# vae.add_loss(vae_loss)

# vae.compile(optimizer='rmsprop', loss=None)
# vae.summary()

# train the VAE on MNIST digits
#(x_train, _), (x_test, y_test) = fashion_mnist.load_data()
#(x_train, _), (x_test, y_test) = mnist.load_data()

#(x_train, _), (x_test, y_test) = cifar10.load_data()
#x_train = x_train.astype('float32') / 255.
#x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
#x_test = x_test.astype('float32') / 255.
#x_test = x_test.reshape((x_test.shape[0],) + original_img_size)


#train_datagen = ImageDataGenerator(
                #rescale = 1./255,
                #horizontal_flip=True)
#test_datagen = ImageDataGenerator(
                #rescale = 1./255,
                #horizontal_flip=True)

#train_generator = train_datagen.flow_from_directory(
                #'dataset/train',
                #target_size = (img_rows, img_cols),
		#color_mode = 'rgb',
                #batch_size = batch_size,
                #class_mode='binary')
#test_generator = test_datagen.flow_from_directory(
                #'dataset/test',
                #target_size = (img_rows, img_cols),
		#color_mode = 'rgb',
                #batch_size = batch_size,
                #class_mode='binary')


#mat = loadmat('cifar_mat.mat')
#mat = loadmat('cedar_mat.mat')
#mat = loadmat('neuro_mat.mat')
#mat = hdf5storage.loadmat('neuro_mat.mat')
mat = hdf5storage.loadmat('neuro_enhanced.mat')
x_train = mat['x_train']
y_train = mat['y_train']
x_test = mat['x_test']
y_test = mat['y_test']


#(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # TEMP


x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
#x_train = x_train.reshape((x_train.shape[0],) +  (img_chns, img_rows, img_cols))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
#x_test = x_test.reshape((x_test.shape[0],) +  (img_chns, img_rows, img_cols))

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print( 'x_train: ', x_train.shape)
print( 'x_test: ', x_test.shape)
print( 'y_train: ', y_train.shape)
print( 'y_test: ', y_test.shape)


#history = vae.fit_generator(train_generator, 
		#steps_per_epoch= 200,
		#epochs = epochs,
		#validation_data= test_generator,
		#validation_steps= 50)

#print('x_train.shape:', x_train.shape)
#print('x_test.shape:', x_test.shape)

#x_train,y_train = train_generator.next()
#x_test,y_test= test_generator.next()


history=vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

#vae.fit(x_train, shuffle=True, epochs=epochs, batch_size = batch_size)

#for e in range(epochs):
	#print('Epoch', e)
	#batches = 0
	#x_train, y_train = train_generator.next()
	#x_test, y_test = test_generator.next()
	#history=vae.fit(x_train,
        	#shuffle=True,
        	#epochs=epochs,
        	#batch_size=batch_size,
        	#validation_data=(x_test, None))
	#encoder = Model(x, z_mean)
	#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
	
	#batches += 1

	#if batches >= batch_size:
		#break


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#x_test_encoded = encoder.predict_generator(test_generator, batch_size=batch_size)



fig = plt.figure()
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.savefig('I_test_encoded.png')
plt.close(fig)

#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()


x_train_encoded = encoder.predict(x_train, batch_size=batch_size)
fig = plt.figure()
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c= y_train)
plt.colorbar()
plt.savefig('I_train_encoded.png')
plt.close(fig)




scipy.io.savemat('./result_VAE_NEURO.mat', mdict={'x_test_encoded': x_test_encoded, 'y_test': y_test,
                'x_train_encoded': x_train_encoded, 'y_train': y_train})

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


vae.save('./models/test_%d_conv_%d_id_%d_e_%d_vae.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
encoder.save('./models/test_%d_conv_%d_id_%d_e_%d_encoder.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
generator.save('./models/test_%d_conv_%d_id_%d_e_%d_generator.h5' % (latent_dim, num_conv, intermediate_dim, epochs))


# save training history
fname = './models/test_%d_conv_%d_id_%d_e_%d_history.pkl' % (latent_dim, num_conv, intermediate_dim, epochs)

with open(fname, 'wb') as file_pi:
	cPickle.dump(history.history, file_pi)

## display a 2D manifold of the digits
n = 40 #15  # figure with 15x15 digits
img_size = img_rows
#figure = np.zeros((digit_size * n, digit_size * n))
#figure = np.zeros((img_size * n, img_size * n, img_chns))
figure = np.zeros((img_size * n, img_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


for i in range(n):
	for j in range(n):
		z_sample = np.array([np.random.uniform(-1.5,1.5, size=latent_dim)])
		x_decoded = generator.predict(z_sample)
		img = x_decoded[0].reshape(img_size, img_size)
		#r = img[0].reshape(img_size,img_size)
		#g = img[1].reshape(img_size,img_size)
		#b = img[2].reshape(img_size,img_size)
		#img2 = cv2.merge([r,g,b])
		figure[ i* img_size: (i+1)*img_size, j*img_size:(j+1)*img_size] = img

#for i, yi in enumerate(grid_x):
	#for j, xi in enumerate(grid_y):
		#z_sample = np.array([[xi, yi]])
		##z_sample = np.tile(z_sample, batch_size).reshape(batch_size,2)
		#x_decoded = generator.predict(z_sample) #, batch_size=batch_size)
		#img = x_decoded[0].reshape(img_chns, img_size, img_size)
                #r = img[0].reshape(img_size,img_size)
                #g = img[1].reshape(img_size,img_size)
                #b = img[2].reshape(img_size,img_size)
                #img2 = cv2.merge([r,g,b])
                #figure[ i* img_size: (i+1)*img_size, 
				#j*img_size:(j+1)*img_size] = img2


fig = plt.figure()
plt.figure(figsize=(30, 30))
plt.imshow(figure, cmap='Greys_r')
plt.imshow(figure)
plt.savefig('I_latent2.png')
plt.close(fig)
#plt.show()
