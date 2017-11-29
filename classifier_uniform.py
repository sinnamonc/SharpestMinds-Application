# This model trained on 10000 batches of 7000 total datapoints (1000 
# sampled uniformly from each class) produces an accuracy of about 77%
# on a similarly sampled batch of test data. Addressing the issue of
# how to sample the training, test, and validation data to deal with
# the uneven distribution of labels seems to be a main difficulty.
# Sampling uniformly at random from each class for all of the data sets
# produces good results. 

# I use the following model in this example:
# model = Sequential()
# model.add(Dense(120, activation='relu', input_dim=54))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(7, activation='softmax'))
# 
# optimizer = RMSprop(lr=0.05)
# model.compile(loss='mean_squared_error', optimizer=optimizer)

import numpy as np
import pandas as pd

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from data_preprocessing import import_data
from utils import shuffle_in_unison
from utils import percent_correct
from utils import get_uniform_batch

# Import the data as a dataframe
df = import_data()


# Create test, validation, and training sets
test_df = pd.DataFrame()
valid_df = pd.DataFrame()
train_df = pd.DataFrame()

train_df_size_per_index = 1620#1620
valid_df_size_per_index = 540#540

#Shuffle the dataframe df
df = df.sample(frac=1)

for i in np.arange(1,8):
    # Extract the rows with label i
    data_temp = df[df.iloc[:,-1] == i]
    # Put the first test_df_size into the test set
    train_df = pd.concat([train_df, data_temp[:train_df_size_per_index]],axis=0, join='outer')
    # Put the next valid_df_size into the validation set
    valid_df = pd.concat([valid_df,data_temp[train_df_size_per_index:train_df_size_per_index+valid_df_size_per_index]],axis=0)
    # Put the remainder into the training set
    test_df = pd.concat([test_df,data_temp[train_df_size_per_index+valid_df_size_per_index:]],axis=0)

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

# Build the model
print('Build model...')
model = Sequential()
model.add(Dense(120, activation='relu', input_dim=54))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


optimizer = RMSprop(lr=0.05)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Initializations
num_epochs_trained = 0
train_loss = []
valid_loss = []

# train the model several epochs, and test on the test set. Plot the loss for train and test sets

for _ in np.arange(10000):
    #print('Creating batch number', num_epochs_trained + 1, '...')
    batch_data, batch_labels = get_uniform_batch(train_data,train_labels)
    print('Training on batch number', num_epochs_trained + 1, '...')
    train_loss.append(model.train_on_batch(batch_data, batch_labels))
    valid_loss.append(model.test_on_batch(valid_data, valid_labels, sample_weight=None))
    print('train_loss =', train_loss[-1], '    valid_loss =', valid_loss[-1])
    num_epochs_trained = num_epochs_trained + 1
print('Total number of epochs trained = {}'.format(num_epochs_trained))

plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Show results in percentage accuracy
percent_correct(model,train_data,train_labels)
percent_correct(model,valid_data,valid_labels)
# Show result on a batch of test data in percentage accuracy
batch_data, batch_labels = get_uniform_batch(test_data,test_labels)
test_loss = model.test_on_batch(batch_data, batch_labels, sample_weight=None)
print(test_loss)
percent_correct(model,batch_data,batch_labels)
