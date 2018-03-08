from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np

# Captalize for Layer

Encode = Dense(32, activation='relu')
Decode = Dense(784, activation='sigmoid')

x = Input(shape=(784,))
z = Encode(x)
y = Decode(z)

autoencoder = Model(x, y)
encoder = Model(x, z)

# I wish I can use any tensor... (other than Input)
#    Model should just figure out it is first layer and get it's output shape
# so I can remove below 2 lines
z = Input(shape=(32,))
y = Decode(z)
decoder = Model(z, y)

'''
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
'''
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
				
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import cv2
import os
import time

def to_png(z):
	z = z*255
	z = z.astype(int)
	z = z.reshape((28,28,1))
	return z

outdir = 'o/{}'.format(int(time.time()))
os.makedirs(outdir)
n = 10  # how many digits we will display
for i in range(n):
    cv2.imwrite(outdir+'/'+str(i)+'_i.png',to_png(x_test[i]))
    cv2.imwrite(outdir+'/'+str(i)+'_o.png',to_png(decoded_imgs[i]))