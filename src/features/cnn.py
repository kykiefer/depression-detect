from __future__ import print_function
import numpy as np
from random_sampling import build_array_of_random_samples
from sklearn.model_selection import train_test_split
np.random.seed(15)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

"""
CNN to classify spectrograms of normal particpants (0) or depressed particpants (1).
Using Theano with TensorFlow image_dim_ordering :
(# images, # rows, # cols, # channels)
(3040, 513, 125, 1) for the X images below
"""


def train_test(X, y, nb_classes, test_size=0.2):
    """
    Split the X, y datasets into training and test sets based on desired test size.

    Parameters
    ----------
    X : array
        X features (represented by spectrogram matrix)
    y : array
        y labels (1 for depressed; 0 for normal)
    nb_classes : int
        number of classes being classified (2 for a binary label)
    test_size : float
        perecentge of data to include in test set

    Returns
    -------
    X_train : array
    X_test : array
    Y_train : array
    Y_test : array
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train, X_test = preprocess(X_train, X_test)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def keras_img_prep(X_train, X_test, img_dep, img_rows, img_cols):
    ''' Reshape feature matrices X_train, X_test to be compatible with neural
      network (NN) expected input_shape.
    if dim_ordering is th (Theano), NN expects:
        (# images, # channels, # rows, # cols)
    if dim_ordering is tf (TensorFlow), NN expects:
        (# images, # rows, # cols, # channels)
    The tf and th image_dim_ordering is first set in the
        ~/keras/keras.json file.  Make sure it matches the
        K.set_image_dim_ordering at the beginning of this script.
    Input: X_train, X_test - arrays
           img_dep, img_rows, img_cols - integers
           dim_ordering - string
    Output: X_train, X_test - arrays
            input_shape - tuple
    '''
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return X_train, X_test, input_shape


def preprocess(X_train, X_test):
    # convert from float64 to float32 for speed
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize here

    return X_train, X_test

def cnn(X_train, y_train, X_test, y_test, kernel_size, pool_size, batch_size, nb_classes, epochs, input_shape):
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    # get X and y arrays
    X, y = build_array_of_random_samples('/Users/ky/Desktop/depression-detect/data/processed')

    # CNN parameters
    batch_size = 228
    nb_classes = 2
    epochs = 3
    kernel_size = (3, 3)
    pool_size = (2, 2)
    nb_filters = 32

    # train/test split for cross validation
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test(X, y, 2, test_size=test_size)

    # specify image dimensions - 513x125x1 for spectrogram with crop size of 125 pixels
    img_rows, img_cols, img_depth = X_train.shape[1], X_train.shape[2], 1

    # prep image input for Keras
    # used TensorFlow dim_ordering (tf), (# images, # rows, # cols, # chans)
    # used Theano backend, tf dim_ordering
    X_train, X_test, input_shape = keras_img_prep(X_train, X_test, img_depth, img_rows, img_cols)

    # fit and score model
    cnn(X_train, y_train, X_test, y_test, kernel_size, pool_size, batch_size, nb_classes, epochs, input_shape)
