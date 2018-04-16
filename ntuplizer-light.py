import itertools
import numpy as np
import openpyxl
import pandas as pd
import time
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D

#listproduct=list(itertools.product('ACGT', repeat=2))
#new_data = list(''.join(w) for w in listproduct)
new_data=['A/A', 'A/C', 'A/G', 'A/T', 'C/A', 'C/C', 'C/G', 'C/T', 'G/A', 'G/C', 'G/G', 'G/T', 'T/A', 'T/C', 'T/G', 'T/T','---']



# Dictionary mapping unique characters to their index in `chars`
chars = sorted(list(set(new_data)))
print(chars)


char_indices = dict((char, chars.index(char)) for char in chars)

#Read the excel file with the genetic markers and the milk index

excel='SNP_cleaned.xlsx'
xmarcadores=pd.read_excel(excel,sheet_name='marcadores')
xleche=pd.read_excel(excel,sheet_name='leche')

#Number of rows = number of basis pairs
maxlon=xmarcadores.shape[0]

#Identification for each cow
headers=xmarcadores.columns.values

x = np.zeros( (len(xmarcadores.columns), maxlon, len(chars)), dtype=np.int)
y = np.zeros(len(xmarcadores.columns), dtype=np.double)

print (time.strftime("%H:%M:%S"))

print('Vectorizacion...')
for i,marcador in enumerate(headers):
	y[i]=xleche[marcador]
	for j, cromosoma in enumerate(xmarcadores[marcador].values):
		x[i, j, char_indices[cromosoma]]=1

print("len of x ", x.shape)
print("len of y ", len(y))

from keras import layers
from keras import models

nb_filter = 256
dense_outputs = 1024
filter_kernels = [7, 7, 3, 3, 3, 3]
n_out = 1
batch_size = 80
nb_epoch = 10

inputs = Input(shape=(maxlon, len(chars)), name='input', dtype='float32')

conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                     border_mode='valid', activation='relu',
                     input_shape=(maxlon, len(chars)))(inputs)
conv = MaxPooling1D(pool_length=3)(conv)

conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                      border_mode='valid', activation='relu')(conv)
conv1 = MaxPooling1D(pool_length=3)(conv1)

conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                      border_mode='valid', activation='relu')(conv1)

conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                      border_mode='valid', activation='relu')(conv2)

conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                      border_mode='valid', activation='relu')(conv3)

conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                      border_mode='valid', activation='relu')(conv4)
conv5 = MaxPooling1D(pool_length=3)(conv5)
conv5 = Flatten()(conv5)

z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

pred = Dense(n_out, name='output')(z)

model = Model(input=inputs, output=pred)

model.compile(loss='mean_absolute_error', optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x, y, batch_size=32, nb_epoch=120, validation_split=0.2, verbose=True)


model.summary()

print(x.shape)
print(y)

		
import time
print (time.strftime("%H:%M:%S"))	




		
from keras import optimizers

model.compile(loss='mean_absolute_error',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

x.shape


history=model.fit(x, y, validation_split=0.1,batch_size=100, epochs=10000)


test=np.array([[[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]])
print("test.shape --> ", test.shape)
print("Value test : ", model.predict(x)[0])
#print("Value test : ", model.predict(test)[0])
