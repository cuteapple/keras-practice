from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model, Sequential
from keras.datasets import mnist
import keras.regularizers as regularizers
import numpy as np

import types
options = types.SimpleNamespace()
del types

options.load = True
options.train = True
options.epochs = 5
options.save = True
options.filename = 'model.h5'

#
# Prepare Data
#

(x_train, _), (x_test, _) = mnist.load_data()
#grayscale 0-255
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# reshape to (sample_count,flatten)
x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784)

print(x_train.shape) #(?,784)
print(x_test.shape) #(?,784)

#
# Prepare Network
#

Encode = Dense(32, activation='relu',activity_regularizer=regularizers.l1(10e-5))
Decode = Dense(784, activation='sigmoid')

x = Input(shape=(784,))
z = Encode(x)
y = Decode(z)
z2 = Input(shape=(32,)) # wish I can use z directly o.o
y2 = Decode(z2) # reuse layer (and it's weights)

autoencoder = Model(x, y)
encoder = Model(x, z)
decoder = Model(z2, y2)

del x,z,y,z2,y2

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#
#
#

if options.load:
	try:
		print('loading {}'.format(options.filename))
		autoencoder.load_weights(options.filename)
	except OSError:
		print('load weights failed')

if options.train:
	autoencoder.fit(
		x_train, x_train,
		epochs = options.epochs,
		batch_size=256,
		shuffle=True,
		validation_data=(x_test, x_test))

if options.save:
	print('saving {}'.format(options.filename))
	autoencoder.save_weights(options.filename)



#
# see result
#

# encode and decode test set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import cv2
import os
import time

def to_img(z):
	z = z*255
	z = z.astype(int)
	z = z.reshape((28,28,1))
	return z

outdir = 'o/{}'.format(int(time.time()))
os.makedirs(outdir)
outdir_for = lambda name,postfix='',extension='png': '{}/{}_{}.{}'.format(outdir,name,postfix,extension)

for i in range(10):
    cv2.imwrite(outdir_for(i,'i'),to_img(x_test[i]))
    cv2.imwrite(outdir_for(i,'o'),to_img(decoded_imgs[i]))