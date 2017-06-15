import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


"""
Plots of test/train accuracy, loss, ROC curve.
"""


def plot_accuracy(history, model_id):
    """
    Plots train and test accuracy for each epoch.
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../../images/cnn{}_accuracy.png'.format(model_id))
    plt.close()


def plot_loss(history, model_id):
    """
    Plots train and test loss for each epoch.
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../../images/cnn{}_loss.png'.format(model_id))
    plt.close()


def plot_roc_curve(y_test, y_score, model_id):
    """
    Plots ROC curve for final trained model. Code taken from:
    https://vkolachalama.blogspot.com/2016/05/keras-implementation-of-mlp-neural.html
    """
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig('../../images/cnn{}_roc.png'.format(model_id))
    plt.close()
