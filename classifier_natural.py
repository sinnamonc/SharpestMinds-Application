import numpy as np
import pandas as pd
import time
# import math
# import sys
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
# from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from data_preprocessing import import_data
from utils import shuffle_in_unison
# from utils import percent_correct
# from utils import get_uniform_batch

print('Package Imports Complete')

# Import the data as a dataframe
df = import_data()
# print('Dataframe shape:',df.shape)
print('Data Imported')


# Create test, validation, and training sets
test_df = pd.DataFrame()
valid_df = pd.DataFrame()
train_df = pd.DataFrame()

# take about 80% of the data for the training and validation sets
train_df_size_per_index = 370000 # about 64% of the data
valid_df_size_per_index = 100000 # about 16% of the data

#Shuffle the dataframe df
df = df.sample(frac=1)

# Put the first test_df_size into the test set
train_df = df[:train_df_size_per_index]
# Put the next valid_df_size into the validation set
valid_df = df[train_df_size_per_index:train_df_size_per_index+valid_df_size_per_index]
# Put the remainder into the training set
test_df = df[train_df_size_per_index+valid_df_size_per_index:]

# Extract the last columns, which corresponds to the labels
test_labels = test_df.iloc[:,-1]
valid_labels = valid_df.iloc[:,-1]
train_labels = train_df.iloc[:,-1]

# Remove the last columns, which corresponds to the labels
test_df = test_df.drop(test_df.columns[-1],axis=1)
valid_df = valid_df.drop(valid_df.columns[-1],axis=1)
train_df = train_df.drop(train_df.columns[-1],axis=1)

# Convert data from dataframes to np.arrays
test_data = test_df.values
valid_data = valid_df.values
train_data = train_df.values
test_labels = test_labels.values
valid_labels = valid_labels.values
train_labels = train_labels.values

# Convert labels to one hot vectors
test_labels = to_categorical(test_labels-1,7)
valid_labels = to_categorical(valid_labels-1,7)
train_labels = to_categorical(train_labels-1,7)

# Shuffle the data and labels
shuffle_in_unison(test_data, test_labels)
shuffle_in_unison(valid_data, valid_labels)
shuffle_in_unison(train_data, train_labels)

print('Data sets created')

# Build the model

model = Sequential()
model.add(Dense(120, activation='relu', input_dim=54))
model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

optimizer = RMSprop(lr=0.05)
# model.compile(loss='mean_squared_error', optimizer=optimizer)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])

# checkpoint
filepath="models/classifier_natural_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('Model Built')
print('Training...')
# training
hist = model.fit(train_data, train_labels, batch_size = 1000, epochs = 20, initial_epoch = 0, verbose = 0, 
              validation_data = (valid_data, valid_labels), callbacks = callbacks_list)

# Plot results
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print('Complete')