from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def aggregate_preds(model, y_test, y_test_pred, y_test_pred_proba):
    # Converting 1-D arrays representing depression
    y_test_pred_proba = y_test_pred_proba[:,1]


    # should put this as a class variable so the 40 isn't hard coded
    n_splits = int(len(y_test) / 40)

    # explicit until we incorporate into class
    partic_truth = [int(partic_clips.mean()) for partic_clips in np.split(y_test, n_splits)]

    # split y_test and and y_test pred into particpants chunks
    partic_proba = np.split(y_test_pred_proba, n_splits)

    for partic_preds in partic_proba:
        plt.xlim([0.0, 1.05])
        plt.hist(partic_pred, bins=20, range=(0,1), normed=True)


    # average all 40 clips
    best_prediction = [partic_clips.mean() for partic_clips in partic_proba]

    # for partic in partic_proba, calculate predict based on compiling 1 through



    # Computing confusion matrix for test dataset
    conf_matrix = confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, conf_matrix

if __name__ == '__main__':
    # load saved model for eval
    model = load_model('/Users/ky/Desktop/depression-detect/src/models/cnn_3_final.h5')

    X_test = np.load('/Users/ky/Desktop/depression-detect/data/processed/test_samples.npz')['arr_0']

    y_test = np.load('/Users/ky/Desktop/depression-detect/data/processed/test_labels.npz')['arr_0']

    X_test = X_test.astype('float32')
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    X_test = X_test.reshape(X_test.shape[0], 1, 513, 125)

    y_test_pred = model.predict_classes(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
