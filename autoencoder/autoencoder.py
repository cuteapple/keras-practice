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
# so I can remove below 2(3) lines
z = Input(shape=(32,))
# reuse layer (and it's weights)
y = Decode(z)

decoder = Model(z, y)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
#grayscale 0-255
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# reshape to (sample_count,flatten)
# well, in face we know (and need) it's 784 since we define Input layer as such
# the following 2 line do the same reshape
x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape) #(?,784)
print(x_test.shape) #(?,784)

#in () python ignore all space like other language
autoencoder.fit(
	x_train, x_train,
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
outdir_for = lambda name,postfix='',extension='png': '{}/{}_{}.{}'.format(outdir,filename,postfix,extension)

#no need for it, it just a constant, maybe change it when need, but not now
#n=10
for i in range(10):
    cv2.imwrite(outdir+'/'+str(i)+'_i.png',to_png(x_test[i]))
    cv2.imwrite(outdir+'/'+str(i)+'_o.png',to_png(decoded_imgs[i]))