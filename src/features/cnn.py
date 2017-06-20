from __future__ import print_function
import boto
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from plot_metrics import plot_accuracy, plot_loss, plot_roc_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
np.random.seed(15)  # for reproducibility


"""
CNN used to classify spectrograms of normal participants (0) or depressed
participants (1). Using Theano backend and Theano image_dim_ordering:
(# channels, # images, # rows, # cols)
(1, 3040, 513, 125)
"""


def retrieve_from_bucket(file):
    """
    Download spectrogram representation of matrices from S3 bucket.
    """
    conn = boto.connect_s3(access_key, access_secret_key)
    bucket = conn.get_bucket('depression-detect')
    file_key = bucket.get_key(file)
    file_key.get_contents_to_filename(file)
    X = np.load(file)
    return X


def preprocess(X_train, X_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    return X_train, X_test


def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))

    # normalize to dBfS
    X_train, X_test = preprocess(X_train, X_test)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def keras_img_prep(X_train, X_test, img_dep, img_rows, img_cols):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return X_train, X_test, input_shape


def cnn(X_train, y_train, X_test, y_test, batch_size,
        nb_classes, epochs, input_shape):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid', strides=1,
                     input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))

    model.add(Conv2D(32, (1, 3), padding='valid', strides=1,
              input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test))

    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return model, history


def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, \
        y_test_pred_proba, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])


def save_to_bucket(file, obj_name):
    """
    Saves local file to S3 bucket for redundancy and repreoducibility
    by others.
    """
    conn = boto.connect_s3(access_key, access_secret_key)

    bucket = conn.get_bucket('depression-detect')

    file_object = bucket.new_key(obj_name)
    file_object.set_contents_from_filename(file)


if __name__ == '__main__':
    model_id = input("Enter model id: ")

    # Load from S3 bucket
    print('Retrieving from S3...')
    X_train = retrieve_from_bucket('train_samples.npz')
    y_train = retrieve_from_bucket('train_labels.npz')
    X_test = retrieve_from_bucket('test_samples.npz')
    y_test = retrieve_from_bucket('test_labels.npz')

    X_train, y_train, X_test, y_test = \
        X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # CNN parameters
    batch_size = 32
    nb_classes = 2
    epochs = 7

    # normalalize data and prep for Keras
    print('Processing images for Keras...')
    X_train, X_test, y_train, y_test = prep_train_test(X_train, y_train,
                                                       X_test, y_test,
                                                       nb_classes=nb_classes)

    # 513x125x1 for spectrogram with crop size of 125 pixels
    img_rows, img_cols, img_depth = X_train.shape[1], X_train.shape[2], 1

    # reshape image input for Keras
    # used Theano dim_ordering (th), (# chans, # images, # rows, # cols)
    X_train, X_test, input_shape = keras_img_prep(X_train, X_test, img_depth,
                                                  img_rows, img_cols)

    # run CNN
    print('Fitting model...')
    model, history = cnn(X_train, y_train, X_test, y_test, batch_size,
                         nb_classes, epochs, input_shape)

    # evaluate model
    print('Evaluating model...')
    y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, \
        conf_matrix = model_performance(model, X_train, X_test, y_train, y_test)

    # save model to locally
    print('Saving model locally...')
    model_name = '../models/cnn_{}.h5'.format(model_id)
    model.save(model_name)

    # custom evaluation metrics
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))

    # plot train/test loss and accuracy. saves files in working dir
    print('Saving plots...')
    plot_loss(history, model_id)
    plot_accuracy(history, model_id)
    plot_roc_curve(y_test[:, 1], y_test_pred_proba[:, 1], model_id)

    # save model S3
    print('Saving model to S3...')
    save_to_bucket(model_name, 'cnn_{}.h5'.format(model_id))
