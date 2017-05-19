from __future__ import print_function
import numpy as np
np.random.seed(15)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


    # manually doing this for now
    X_train = depressed_samples[319] + depressed_samples[320] + normal_samples[302] + normal_samples[303]
    X_test = depressed_samples[321] + normal_samples[304]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.concatenate((np.ones(len(depressed_samples[319])), np.ones(len(depressed_samples[320])), np.zeros(len(depressed_samples[320])), np.zeros(len(normal_samples[303]))), axis=0)

    y_test = np.concatenate((np.ones(len(depressed_samples[321])), np.ones(len(normal_samples[304]))), axis=0)




'''Trains a simple convnet on the MNIST dataset.
based on a keras example by fchollet
Find a way to improve the test accuracy to almost 99%!
FYI, the number of layers and what they do is fine.
But their parameters and other hyperparameters could use some work.
'''

# batch_size = 1000 # number of training samples used at a time to update the weights
# nb_classes = 2  # number of output possibilites: [0 - 9] (don't change)
# nb_epoch = 2    # number of passes through the entire train dataset before weights "final"
#
# # input image dimensions
# img_rows, img_cols = 513, 125
# # number of convolutional filters to use
# nb_filters = 12
# # size of pooling area for max pooling
# pool_size = (2, 2) # decreases image size, and helps to avoid overfitting
# # convolution kernel size
# kernel_size = (4, 4) # slides over image to learn features
#
# # the data, shuffled and split between train and test sets
# # (X_train, y_train), (X_test, y_test) = mnist.load_data()
# input_shape = (img_rows, img_cols, 1)
#
#
# # # don't change conversion or normalization
# # X_train = X_train.astype('float32') # data was uint8 [0-255]
# # X_test = X_test.astype('float32')  # data was uint8 [0-255]
# # X_train /= 255 # normalizing (scaling from 0 to 1)
# # X_test /= 255  # normalizing (scaling from 0 to 1)
# # print('X_train shape:', X_train.shape)
# # print(X_train.shape[0], 'train samples')
# # print(X_test.shape[0], 'test samples')
# #
# # convert class vectors to binary class matrices (don't change)
# Y_train = np_utils.to_categorical(y_train, nb_classes) # cool
# Y_test = np_utils.to_categorical(y_test, nb_classes)   # cool * 2
# # in Ipython you should compare Y_test to y_test
#
# model = Sequential() # model is a linear stack of layers (don't change)
#
# # note: the convolutional layers and dense layers require an activation function
# # see https://keras.io/activations/
# # and https://en.wikipedia.org/wiki/Activation_function
# # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
#
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=input_shape)) #first conv. layer (keep layer)
# model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers
#
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1])) #2nd conv. layer (keep layer)
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
# model.add(Dropout(0.2)) # zeros out some fraction of inputs, helps prevent overfitting
#
# model.add(Flatten()) # necessary to flatten before going into conventional dense layer (keep layer)
# print('Model flattened out to ', model.output_shape)
#
# # now start a typical neural network
# model.add(Dense(128)) # (only) 32 neurons in this layer, really?  (keep layer)
# model.add(Activation('tanh'))
#
# model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting
#
# model.add(Dense(nb_classes)) # 10 final nodes (one for each class) (keep layer)
# model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-9
#
# # many optimizers available
# # see https://keras.io/optimizers/#usage-of-optimizers
# # suggest you keep loss at 'categorical_crossentropy' for this multiclass problem,
# # and metrics at 'accuracy'
# # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
# # how are we going to solve and evaluate it:
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])
#
# # during fit process watch train and test error simultaneously
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))
#
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1]) # this is the one we care about

if __name__ == '__main__':
    depressed_samples, normal_samples = create_sample_dicts()
    test_train_split(depressed_samples, normal_samples, test_size=.25)
