import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score


"""
Plots of test/train accuracy, loss, roc curve
"""


def plot_accuracy(history, model_id):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_accuracy.png'.format(model_id))


def plot_loss(history, model_id):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_loss.png'.format(model_id))


def plot_roc_curve(history, model_id):
    pass
