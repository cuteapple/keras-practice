from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np

import types
options = types.SimpleNamespace()
del types

options.load = True
options.train = True
options.epochs = 20
options.save = True
options.test = False
options.plot_model = False
options.filename = 'model.h5'

#
# Prepare Data
#

if options.train or options.test:
	(x_train, _), (x_test, _) = mnist.load_data()
	#grayscale 0-255
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.

	# reshape to (sample_count,flatten)
	x_train = x_train.reshape(x_train.shape[0],28,28,1)
	x_test = x_test.reshape(x_test.shape[0],28,28,1)

	print(x_train.shape) #(?,784)
	print(x_test.shape) #(?,784)

#
# Prepare Network
#

x = Input(shape=(28,28,1))
z = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
z = MaxPooling2D((2, 2), padding='same')(z)
z = Conv2D(8, (3, 3), activation='relu', padding='same')(z)
z = MaxPooling2D((2, 2), padding='same')(z)
z = Conv2D(8, (3, 3), activation='relu', padding='same')(z)
z = MaxPooling2D((2, 2), padding='same')(z)
Encode = Model(x,z,name='Encoder')

z = Input(shape = Encode.output_shape[1:])
y = Conv2D(8, (3, 3), activation='relu', padding='same')(z)
y = UpSampling2D((2, 2))(y)
y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)
y = UpSampling2D((2, 2))(y)
y = Conv2D(16, (3, 3), activation='relu')(y)
y = UpSampling2D((2, 2))(y)
y = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
Decode = Model(z,y,name='Decoder')

x = Input(shape=(28,28,1))
z = Encode(x)
y = Decode(z)

autoencoder = Model(x,y)
encoder = Encode
decoder = Decode

if options.plot_model:
	from keras.utils import plot_model
	plot_model(autoencoder, to_file='model.png',show_shapes=True)
	plot_model(encoder, to_file='model_encoder.png',show_shapes=True)
	plot_model(decoder, to_file='model_decoder.png',show_shapes=True)



#
# Start
#

if options.load:
	try:
		print('loading {}'.format(options.filename))
		autoencoder.load_weights(options.filename)
	except (OSError,ValueError) as e:
		print(str(e))
		print('load weights failed, recreate')

if options.train:
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
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

if options.test:
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