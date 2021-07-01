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
    parser.add_argument('-b', '--Background', help = 'Background: \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\'', type = str, default = 'all')
    parser.add_argument('-t', '--TrainingFraction', help = 'Relative size of the training sample, between 0 and 1', default = 0.8)
    parser.add_argument('-p', '--PreselectionCuts', help = 'Preselection cut', type = str)
    parser.add_argument('-n', '--Nodes', help = 'Number of nodes of the (p)DNN, should always be >= nColumns and strictly positive', default = 32)
    parser.add_argument('-l', '--Layers', help = 'Number of layers of the (p)DNN', default = 2)
    parser.add_argument('-e', '--Epochs', help = 'Number of epochs for the training', default = 150)
    parser.add_argument('-v', '--Validation', help = 'Fraction of the training data that will actually be used for validation', default = 0.2)
    parser.add_argument('-d', '--Dropout', help = 'Fraction of the neurons to drop during the training', default = 0.2)
    parser.add_argument('-m', '--Mass', help = 'Mass to analyze in the DNN')
    
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
    background = args.Background.split()
    if args.Background is None:
        parser.error(Fore.RED + 'Requested background (\'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\'')
    backgroundString = 'all'
    if args.Background != 'all':
        backgroundString = '_'.join([str(item) for item in background]) ### altro?
    trainingFraction = float(args.TrainingFraction) ### altro?
    if args.TrainingFraction and (trainingFraction < 0. or trainingFraction > 1.):
        parser.error(Fore.RED + 'Training fraction must be between 0 and 1')
    preselectionCuts = args.PreselectionCuts
    if args.PreselectionCuts is None:
        preselectionCuts = 'none'
    numberOfNodes = int(args.Nodes)
    if args.Nodes and numberOfNodes < 1:
        parser.error(Fore.RED + 'Number of nodes must be integer and strictly positive')
    numberOfLayers = int(args.Layers)
    if args.Layers and numberOfLayers < 1:
        parser.error(Fore.RED + 'Number of layers must be integer and strictly positive')
    numberOfEpochs = int(args.Epochs)
    if args.Epochs and numberOfEpochs < 1:
        parser.error(Fore.RED + 'Number of epochs must be integer and strictly positive')
    validationFraction = float(args.Validation)
    if args.Validation and (validationFraction < 0. or validationFraction > 1.):
        parser.error(Fore.RED + 'Validation fraction must be between 0 and 1')
    dropout = float(args.Dropout)
    if args.Dropout and (dropout < 0. or dropout > 1.):
        parser.error(Fore.RED + 'Dropout must be between 0 and 1')
    mass = float(args.Mass)

    #if args.Dropout and (dropout < 0. or dropout > 1.):
    #    parser.error(Fore.RED + 'Dropout must be between 0 and 1')

    print(Fore.BLUE + '  training fraction = ' + str(trainingFraction))
    print(Fore.BLUE + '              nodes = ' + str(numberOfNodes))
    print(Fore.BLUE + '             layers = ' + str(numberOfLayers))
    print(Fore.BLUE + '             epochs = ' + str(numberOfEpochs))
    print(Fore.BLUE + 'validation fraction = ' + str(validationFraction))
    print(Fore.BLUE + '            dropout = ' + str(dropout))

    return analysis, channel, signal, jetCollection, backgroundString, trainingFraction, preselectionCuts, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, mass

### Reading from the configuration file
import configparser, ast

def ReadConfig(analysis, jetCollection):
    config = configparser.ConfigParser()
    config.read('Configuration.ini')
    dfPath = config.get('config', 'dfPath')
    dfPath += jetCollection + '/'# + '_DataFrames/'
    #modelPath = config.get('config', 'modelPath')
    #modelPath += jetCollection + '/'
    if analysis == 'merged':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesMerged'))
    elif analysis == 'resolved':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolved'))
    massColumnIndex = InputFeatures.index('mass')
    #return dfPath, modelPath, InputFeatures, massColumnIndex
    return dfPath, InputFeatures, massColumnIndex

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
def LoadData(dfPath, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts):
    #directory = 'OutputDataFrames/' + jetCollection + '/' + signal + '/' + analysis + '/' + channel#dfPath all'inizio
    fileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + signal + '_' + preselectionCuts + '_' + background + '_' + str(trainingFraction) + 't'
    X_Train = np.genfromtxt(dfPath + '/X_train_' + fileCommonName + '.csv', delimiter=',') 
    X_Test = np.genfromtxt(dfPath + '/X_test_' + fileCommonName + '.csv', delimiter=',') 
    y_Train = np.genfromtxt(dfPath + '/y_train_' + fileCommonName + '.csv', delimiter=',') 
    y_Test = np.genfromtxt(dfPath + '/y_test_' + fileCommonName + '.csv', delimiter=',') 
    m_Train_unscaled = np.genfromtxt(dfPath + '/m_train_unscaled_' + fileCommonName + '.csv', delimiter=',') 
    m_Test_unscaled = np.genfromtxt(dfPath + '/m_test_unscaled_' + fileCommonName + '.csv', delimiter=',') 
    X_Input = np.concatenate((X_Train, X_Test), axis = 0)
    return X_Train, X_Test, y_Train, y_Test, m_Train_unscaled, m_Test_unscaled, X_Input

### Writing in the log file
def WritingLogFile(dfPath, X_input, X_test, y_test, X_train, y_train, InputFeatures, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, useWeights):
    logString = 'dfPath: ' + dfPath + '\nNumber of input events: ' + str(X_input.shape[0]) + '\nNumber of test events: ' + str(int(X_test.shape[0])) + ' (' + str(sum(y_test)) + ' signal and ' + str(len(y_test) - sum(y_test)) + ' background)' + '\nNumber of train events: ' + str(X_train.shape[0]) + ' (' + str(sum(y_train)) + ' signal and ' + str(len(y_train) - sum(y_train)) + ' background)' + '\nInputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nuseWeights: ' + str(useWeights)
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
def DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, mass = 0):
    plt.plot(modelMetricsHistory.history['accuracy'])
    plt.plot(modelMetricsHistory.history['val_accuracy'])
    titleAccuracy = NN + ' model accuracy'
    if NN == 'DNN':
        titleAccuracy += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'lower right')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    plt.figtext(0.5, 0.3, legendText, wrap = True, horizontalalignment = 'left')
    #plt.figtext(0.69, 0.28, 'Test accuracy: ' + str(round(testAccuracy, 3)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    AccuracyPltName = outputDir + '/Accuracy.png'
    plt.savefig(AccuracyPltName)
    print('Saved ' + AccuracyPltName)
    plt.clf()

### Drawing Loss
def DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, mass = 0):
    plt.plot(modelMetricsHistory.history['loss'])
    plt.plot(modelMetricsHistory.history['val_loss'])
    titleLoss = NN + ' model loss'
    if NN == 'DNN':
        titleLoss += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleLoss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper right')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\npreselection cuts: ' + PreselectionCuts + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    plt.figtext(0.5, 0.5, legendText, wrap = True, horizontalalignment = 'left')
    #plt.figtext(0.7, 0.7, 'Test loss: ' + str(round(testLoss, 3)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    LossPltName = outputDir + '/Loss.png'
    plt.savefig(LossPltName)
    print('Saved ' + LossPltName)
    plt.clf()

def integral(y,x,bins):
    x_min=x
    s=0
    for i in np.where(bins>x)[0][:-1]:
#        s=s+y[i]*(bins[i+1]-bins[i])
#        print(i,s)
        s=s+y[i]*(bins[i+1]-bins[i])
    return s

### Drawing scores, ROC and efficiency
import numpy as np

def DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, savePlot):

    ### Scores
    Nbins = 1000
    plt.hist(yhat_train_signal, bins = Nbins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal train'], density = True)
    y_signal, bins_1, _ = plt.hist(yhat_test_signal, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal test'], density = True)
    plt.hist(yhat_train_bkg, bins = Nbins, histtype = 'step', lw = 2, color = 'red', label = [r'Background train'], density = True)
    y_bkg, bins_0, _ = plt.hist(yhat_test_bkg, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background test'], density = True)

    if savePlot:
        plt.ylabel('Norm. entries')
        plt.xlabel('Score')
        plt.yscale('log')
        titleScores = NN + ' scores (mass: ' + str(int(mass)) + ' GeV)'
        plt.title(titleScores)
        plt.legend(loc = 'upper center')
        ScoresPltName = outputDir + '/Scores.png'
        plt.savefig(ScoresPltName)
        print('Saved ' + ScoresPltName)
        plt.clf()

    ### ROC
    Nsignal = integral(y_signal, 0, bins_1)
    Nbkg = integral(y_bkg, 0, bins_0)
    signal_eff = np.array([])
    bkg_eff = np.array([])
    y_s = 0
    y_n = 0
    for i in range(0, Nbins + 1):
        x = i/Nbins
        y_s = integral(y_signal, x, bins_1) / Nsignal
        y_n = integral(y_bkg, x, bins_0) / Nbkg
        signal_eff = np.append(y_s, signal_eff)
        bkg_eff = np.append(y_n, bkg_eff)

    Area=round(1000*abs(integral(signal_eff,0,bkg_eff)))/1000
    if savePlot:
        lab='Area: '+str(Area)
        plt.plot(bkg_eff,signal_eff,label=lab,color = 'darkorange', lw = 2)
        #plt.plot([0,1],[0,1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        titleROC = NN + ' ROC curve (mass: ' + str(int(mass)) + ' GeV)'
        plt.title(titleROC)
        legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        plt.figtext(0.5, 0.4, legendText, wrap = True, horizontalalignment = 'left')
        plt.figtext(0.5, 0.25, 'AUC: ' + str(round(Area, 2)), wrap = True, horizontalalignment = 'center')
        ROCPltName = outputDir + '/ROC.png'
        plt.savefig(ROCPltName)
        print('Saved ' + ROCPltName)
        plt.clf()

    ### Background rejection vs efficiency
    WP=[0.90,0.94,0.97,0.99]
    rej=1./bkg_eff
    WP_idx=[np.where(np.abs(signal_eff-WP[i])==np.min(np.abs(signal_eff-WP[i])))[0][0] for i in range(0,len(WP))]
    WP_rej=[str(round(10*rej[WP_idx[i]])/10) for i in range(0,len(WP))]

    if savePlot:
        plt.plot(signal_eff,rej)
        for i in range(0,len(WP)):
            plt.axvline(x=WP[i],color='Red',linestyle='dashed',label='Bkg Rejection @ '+str(WP[i])+' WP: '+WP_rej[i])
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection')
        plt.xlim([0.85,1])
        plt.yscale('log')
        plt.title(NN + ' background rejection curve (mass: ' + str(mass) + ' GeV)')
        EffPltName = outputDir + '/BkgRejection.png'
        plt.savefig(EffPltName)
        print('Saved ' + EffPltName)
        plt.clf()

    return Area, WP, WP_rej 

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
    titleCM = 'Confusion matrix (mass: ' + str(int(mass)) + ' GeV)'
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
    w_train = []
    for event in y_train:
        if event == 0:
            w_train.append(WTrainBkg)
        else:
            w_train.append(WTrainSignal)
    w_train = np.array(w_train)

    return WTrainSignal, WTrainBkg, w_train

