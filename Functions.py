### Checking if the output directory exists. If not, creating it
import os
from colorama import init, Fore
init(autoreset = True)

def checkCreateDir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        return Fore.RED + ' (created)'
    else:
        return Fore.RED + ' (already there)'

### Building the DNN
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation

def BuildDNN(N_input, width, depth):
    model = Sequential()
    model.add(Dense(units = width, input_dim = N_input))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for i in range(0, depth):
        model.add(Dense(width))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

### Drawing Accuracy
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7,7] # Setting plot size
plt.rcParams.update({'font.size': 18}) # Setting font size

def DrawAccuracy(modelMetricsHistory, testAccuracy, titleAccuracy, AccuracyPltName):
    plt.plot(modelMetricsHistory.history['accuracy'])
    plt.plot(modelMetricsHistory.history['val_accuracy'])
    plt.title(titleAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'lower right')
    plt.figtext(0.69, 0.28, 'Test accuracy: ' + str(round(testAccuracy, 3)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    plt.savefig(AccuracyPltName)
    print('Saved ' + AccuracyPltName)
    plt.clf()

### Drawing Loss
def DrawLoss(modelMetricsHistory, testLoss, titleLoss, LossPltName):
    plt.plot(modelMetricsHistory.history['loss'])
    plt.plot(modelMetricsHistory.history['val_loss'])
    plt.title(titleLoss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper right')
    plt.figtext(0.7, 0.7, 'Test loss: ' + str(round(testLoss,2)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    plt.savefig(LossPltName)
    print('Saved ' + LossPltName)
    plt.clf()

### Drawing ROC
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
def DrawROC(fpr, tpr, titleROC, ROCPltName):
    plt.plot(fpr,  tpr, color = 'darkorange', lw = 2, label = 'Full curve')
    plt.plot([0, 0], [1, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(titleROC)
    plt.legend(loc = 'lower right')
    plt.savefig(ROCPltName)
    print('Saved ' + ROCPltName)
    plt.clf()

### Drawing Scores
import numpy as np

def DrawScores(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, titleScores, ScoresPltName):
    bins = np.linspace(0, 1, 40)
    plt.hist(yhat_train_signal, bins = bins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal Train'], density = True)
    plt.hist(yhat_test_signal, bins = bins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal Test'], density = True)
    plt.hist(yhat_train_bkg, bins = bins, histtype = 'step', lw = 2, color = 'red', label = [r'Background Train'], density = True)
    plt.hist(yhat_test_bkg, bins = bins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background Test'], density = True)
    plt.ylabel('Norm. Entries')
    plt.xlabel('DNN score')
    plt.title(titleScores)
    plt.legend(loc = 'upper center')
    plt.savefig(ScoresPltName)
    print('Saved ' + ScoresPltName)
    plt.clf()

### Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def DrawCM(cm, normalize, titleCM, CMPltName):
    classes = ['Background', 'Signal']
    np.set_printoptions(precision = 2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    cmap = plt.cm.Oranges#Blues
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(titleCM)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation = 90)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(CMPltName)
    #plt.show()
    print('Saved ' + CMPltName)
    plt.clf()    
