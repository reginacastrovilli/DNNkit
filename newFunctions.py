### Reading from command line
from argparse import ArgumentParser
from colorama import init, Fore
init(autoreset = True)

def ReadArgParser():
    parser = ArgumentParser()
    parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
    parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
    parser.add_argument('-s', '--Signal', help = 'Signal: \'VBFHVTWZ\', \'Radion\', \'RSG\' or \'VBFRadion\'', type = str, default = 'VBFRadion')
    parser.add_argument('-j', '--JetCollection', help = 'Jet collection: \'TCC\'', type = str, default = 'TCC')
    parser.add_argument('-n', '--Nodes', help = 'Number of nodes of the (p)DNN, should always be >= nColumns and strictly positive', default = 32)
    parser.add_argument('-l', '--Layers', help = 'Number of layers of the (p)DNN', default = 2)
    parser.add_argument('-e', '--Epochs', help = 'Number of epochs for the training', default = 150)
    parser.add_argument('-v', '--Validation', help = 'Fraction of the training data that will actually be used for validation', default = 0.2)
    parser.add_argument('-d', '--Dropout', help = 'Fraction of the neurons to drop during the training', default = 0.2)
    
    args = parser.parse_args()
    
    analysis = args.Analysis
    if args.Analysis is None:
        parser.error(Fore.RED + 'Requested type of analysis (either \'mergered\' or \'resolved\')')
    elif args.Analysis != 'resolved' and args.Analysis != 'merged':
        parser.error(Fore.RED + 'Analysis can be either \'merged\' or \'resolved\'')
    channel = args.Channel
    if args.Channel is None:
        parser.error(Fore.RED + 'Requested channel (either \'ggF\' or \'VBF\')')
    elif args.Channel != 'ggF' and args.Channel != 'VBF':
        parser.error(Fore.RED + 'Channel can be either \'ggF\' or \'VBF\'')
    signal = args.Signal
    if args.Signal is None:
        parser.error(Fore.RED + 'Requested signal (\'VBFHVTWZ\', \'Radion\', \'RSG\' or \'VBFRadion\')')
    elif args.Signal != 'VBFHVTWZ' and args.Signal != 'Radion' and args.Signal != 'RSG' and args.Signal != 'VBFRadion':
        parser.error(Fore.RED + 'Signal can be \'VBFHVTWZ\', \'Radion\', \'RSG\' or \'VBFRadion\'')
    jetCollection = args.JetCollection
    if args.JetCollection is None:
        parser.error(Fore.RED + 'Requested jet collection (\'TCC\' or )')
    elif args.JetCollection != 'TCC':
        parser.error(Fore.RED + 'Jet collection can be \'TCC\', ')
    numberOfNodes = int(args.Nodes)
    if args.Nodes and numberOfNodes < 1:
        parser.error(Fore.RED + 'Number of nodes must be strictly positive')
    numberOfLayers = int(args.Layers)
    if args.Layers and numberOfLayers < 1:
        parser.error(Fore.RED + 'Number of layers must be strictly positive')
    numberOfEpochs = int(args.Epochs)
    if args.Epochs and numberOfEpochs < 1:
        parser.error(Fore.RED + 'Number of epochs must be strictly positive')
    validationFraction = float(args.Validation)
    if args.Validation and (validationFraction < 0. or validationFraction > 1.):
        parser.error(Fore.RED + 'Validation fraction must be between 0 and 1')
    dropout = float(args.Dropout)
    if args.Dropout and (dropout < 0. or dropout > 1.):
        parser.error(Fore.RED + 'Dropout must be between 0 and 1')

    print(Fore.BLUE + '              nodes = ' + str(numberOfNodes))
    print(Fore.BLUE + '             layers = ' + str(numberOfLayers))
    print(Fore.BLUE + '             epochs = ' + str(numberOfEpochs))
    print(Fore.BLUE + 'validation fraction = ' + str(validationFraction))
    print(Fore.BLUE + '            dropout = ' + str(dropout))

    return analysis, channel, signal, jetCollection, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, trainingFraction

### Reading from the configuration file
import configparser, ast

def ReadConfig(analysis, jetCollection):
    config = configparser.ConfigParser()
    config.read('newConfiguration.ini')
    dfPath = config.get('config', 'dfPath')
    dfPath += jetCollection + '_DataFrames/'
    modelPath = config.get('config', 'modelPath')
    modelPath += jetCollection + '/'
    if analysis == 'merged':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesMerged'))
    elif analysis == 'resolved':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolved'))
    return dfPath, modelPath, InputFeatures

### Checking if the output directory exists. If not, creating it
import os

def checkCreateDir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        return Fore.RED + ' (created)'
    else:
        return Fore.RED + ' (already there)'

### Loading input data
import pandas as pd

def LoadData(dfPath, analysis, channel, InputFeatures):
    dfInput = dfPath + 'MixData_PD_' + analysis + '_' + channel + '.pkl'
    df = pd.read_pickle(dfInput)
    X = df[InputFeatures].values
    y = df['isSignal']
    return X, y, dfInput

def newLoadData(dfPath, signal, analysis, channel):
    X_Train = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/X_train_scaled.csv', delimiter=',')
    X_Test = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/X_test_scaled.csv', delimiter=',')
    y_Train = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/y_train.csv')
    y_Test = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/y_test.csv')
    y_Train_cat = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/y_train_cat.csv', delimiter=',')
    y_Test_cat = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/y_test.csv', delimiter=',')
    X_Test_unscaled = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/X_test.csv', delimiter=',')
    X_Train_unscaled = np.genfromtxt(dfPath + '/outDF/' + signal + '_' + analysis + '_' + channel + '_/X_train.csv', delimiter=',')
    X_Input = np.concatenate((X_Train, X_Test), axis = 0)
    return X_Train, X_Test, y_Train, y_Test, y_Train_cat, y_Test_cat, X_Test_unscaled, X_Train_unscaled, X_Input

### Writing in the log file
def WritingLogFile(dfPath, modelPath, X_input, X_test, y_test, X_train, y_train, InputFeatures, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, trainingFraction, useWeights):
    logString = 'dfPath: ' + dfPath + '\nModelPath: ' + modelPath + '\nNumber of input events: ' + str(X_input.shape[0]) + '\nNumber of test events: ' + str(int(X_test.shape[0])) + ' (' + str(sum(y_test)) + ' signal and ' + str(len(y_test) - sum(y_test)) + ' background)' + '\nNumber of train events: ' + str(X_train.shape[0]) + ' (' + str(sum(y_train)) + ' signal and ' + str(len(y_train) - sum(y_train)) + ' background)' + '\nInputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nTraining fraction: ' + str(trainingFraction) + '\nuseWeights: ' + str(useWeights)
    return logString

### Shuffling data
import sklearn.utils

def ShufflingData(df):
    df = sklearn.utils.shuffle(df, random_state = 123)
    #df = df.reset_index(drop = True)
    return df

### Building the (P)DNN
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation

def BuildDNN(N_input, width, depth, dropout):
    model = Sequential()
    model.add(Dense(units = width, input_dim = N_input))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for i in range(0, depth):
        model.add(Dense(width))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))
    #model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model

def SaveArchAndWeights(model, outputDir):
    arch = model.to_json()
    outputArch = outputDir + '/architecture.json'
    with open(outputArch, 'w') as arch_file:
        arch_file.write(arch)
    print('Saved architecture in ' + outputArch)
    outputWeights = outputDir + '/weights.h5'
    model.save_weights(outputWeights)
    print('Saved weights in ' + outputWeights)

def SaveVariables(outputDir, X_input, InputFeatures):
    outputVar = outputDir + '/variables.json'
    with open(outputVar, 'w') as var_file:
        var_file.write("{\n")
        var_file.write("  \"inputs\": [\n")
        for col in range(X_input.shape[1]):
            offset = -1. * float(X_input.mean(axis=0)[col])
            scale = 1. / float(X_input.std(axis=0)[col])
            var_file.write("    {\n")
            var_file.write("      \"name\": \"%s\",\n" % InputFeatures[col])
            var_file.write("      \"offset\": %lf,\n" % offset) # EJS 2021-05-27: I have compelling reasons to believe this should be -mu
            var_file.write("      \"scale\": %lf\n" % scale) # EJS 2021-05-27: I have compelling reasons to believe this should be 1/sigma
            var_file.write("    }")
            if (col < X_input.shape[1]-1):
                var_file.write(",\n")
            else:
                var_file.write("\n")
        var_file.write("  ],\n")
        var_file.write("  \"class_labels\": [\"BinaryClassificationOutputName\"]\n")
        var_file.write("}\n")
    print('Saved variables in ' + outputVar)

def SaveFeatureScaling(outputDir, X_input, InputFeatures):
    outputFeatureScaling = outputDir + '/FeatureScaling.dat'
    with open(outputFeatureScaling, 'w') as scaling_file: # EJS 2021-05-27: check which file name is hardcoded in the CxAODReader
        scaling_file.write("[")
        scaling_file.write(', '.join(str(i) for i in InputFeatures))
        scaling_file.write("]\n")
        scaling_file.write("Mean\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(float(i)) for i in X_input.mean(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("minusMean\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(-float(i)) for i in X_input.mean(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("Var\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(float(i)) for i in X_input.var(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("sqrtVar\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(float(i)) for i in X_input.std(axis=0)))
        scaling_file.write("]\n")
        scaling_file.write("OneOverStd\n")
        scaling_file.write("[")
        scaling_file.write(' '.join(str(1./float(i)) for i in X_input.std(axis=0)))
        scaling_file.write("]\n")
    print('Saved features scaling in ' + outputFeatureScaling)
    
def SaveModel(model, X_input, InputFeatures, outputDir):
    SaveArchAndWeights(model, outputDir)
    SaveVariables(outputDir, X_input, InputFeatures)
    SaveFeatureScaling(outputDir, X_input, InputFeatures)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams.update({'font.size': 16})

### Evaluating the (P)DNN performance
def EvaluatePerformance(model, X_test, y_test):
    perf = model.evaluate(X_test, y_test, batch_size = 2048)
    testLoss = perf[0]
    testAccuracy = perf[1]
    print(format(Fore.BLUE + 'Test loss: ' + str(testLoss)))
    print(format(Fore.BLUE + 'Test  accuracy: ' + str(testAccuracy)))
    
    return testLoss, testAccuracy

### Prediction on signal and background separately
def PredictionSigBkg(model, X_train_signal, X_train_bkg, X_test_signal, X_test_bkg):
    yhat_train_signal = model.predict(X_train_signal, batch_size = 2048)
    yhat_train_bkg = model.predict(X_train_bkg, batch_size = 2048)
    yhat_test_signal = model.predict(X_test_signal, batch_size = 2048)
    yhat_test_bkg = model.predict(X_test_bkg, batch_size = 2048)

    return yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg

### Drawing Accuracy
def DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, mass = 0):
    plt.plot(modelMetricsHistory.history['accuracy'])
    plt.plot(modelMetricsHistory.history['val_accuracy'])
    titleAccuracy = 'Model accuracy'
    if NN == 'DNN':
        titleAccuracy = 'Model accuracy (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'lower right')
    #plt.figtext(0.69, 0.28, 'Test accuracy: ' + str(round(testAccuracy, 3)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    AccuracyPltName = outputDir + '/Accuracy.png'
    plt.savefig(AccuracyPltName)
    print('Saved ' + AccuracyPltName)
    plt.clf()

### Drawing Loss
def DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, mass = 0):
    plt.plot(modelMetricsHistory.history['loss'])
    plt.plot(modelMetricsHistory.history['val_loss'])
    titleLoss = 'Model loss'
    if NN == 'DNN':
        titleLoss = 'Model loss (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleLoss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper right')
    #plt.figtext(0.7, 0.7, 'Test loss: ' + str(round(testLoss, 3)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    LossPltName = outputDir + '/Loss.png'
    plt.savefig(LossPltName)
    print('Saved ' + LossPltName)
    plt.clf()

### Drawing ROC (Receiver Operating Characteristic)
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
def DrawROC(fpr, tpr, AUC, outputDir, mass):
    plt.plot(fpr,  tpr, color = 'darkorange', lw = 2)
    #plt.plot([0, 0], [1, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    titleROC = 'ROC curves (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleROC)
    plt.figtext(0.7, 0.25, 'AUC: ' + str(round(AUC, 2)), wrap = True, horizontalalignment = 'center')
    ROCPltName = outputDir + '/ROC.png'
    plt.savefig(ROCPltName)
    print('Saved ' + ROCPltName)
    plt.clf()

### Drawing Scores
import numpy as np

def DrawScores(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass):
    bins = np.linspace(0, 1, 40)
    plt.hist(yhat_train_signal, bins = bins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal Train'], density = True)
    plt.hist(yhat_test_signal, bins = bins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal Test'], density = True)
    plt.hist(yhat_train_bkg, bins = bins, histtype = 'step', lw = 2, color = 'red', label = [r'Background Train'], density = True)
    plt.hist(yhat_test_bkg, bins = bins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background Test'], density = True)
    plt.ylabel('Norm. Entries')
    plt.xlabel(NN + ' score')
    plt.yscale("log")
    titleScores = 'Scores (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleScores)
    plt.legend(loc = 'upper center')
    ScoresPltName = outputDir + '/Scores.png'
    plt.savefig(ScoresPltName)
    print('Saved ' + ScoresPltName)
    plt.clf()

### Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def DrawCM(yhat_test, y_test, normalize, outputDir, mass):
    yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test])
    cm = confusion_matrix(y_test, yResult_test_cls)
    classes = ['Background', 'Signal']
    np.set_printoptions(precision = 2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    cmap = plt.cm.Oranges#Blues
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    titleCM = 'Confusion matrix (Mass: ' + str(int(mass)) + ' GeV)'
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
    CMPltName = outputDir + '/ConfusionMatrix.png'
    plt.savefig(CMPltName)
    #plt.show()
    print('Saved ' + CMPltName)
    plt.clf()    

def EventsWeight(y_train):
    signalNum = sum(y_train)
    bkgNum = len(y_train) - signalNum
    if signalNum > bkgNum:
        WTrainSignal = bkgNum / signalNum       
        WTrainBkg = 1
    else:
        WTrainBkg = signalNum / bkgNum       
        WTrainSignal = 1

    return WTrainSignal, WTrainBkg

def NewEventsWeight(y_train):
    signalNum = sum(y_train)
    bkgNum = len(y_train) - signalNum
    if signalNum > bkgNum:
        WTrainSignal = bkgNum / signalNum       
        WTrainBkg = 1
    else:
        WTrainBkg = signalNum / bkgNum       
        WTrainSignal = 1
    w_train = []
    for event in y_train:
        if event == 0:
            w_train.append(WTrainSignal)
        else:
            w_train.append(WTrainBkg)
    w_train = np.array(w_train)

    return WTrainSignal, WTrainBkg, w_train

