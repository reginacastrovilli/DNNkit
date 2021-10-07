### Assigning script names to variables
fileName1 = 'saveToPkl.py'
fileName2 = 'buildDataset.py'
fileName3 = 'splitDataset.py'
fileName4 = 'buildDNN.py'
fileName5 = 'buildPDNN.py'
#fileName4 = 'buildPDNNperEnrico.py'
#fileName4 = 'testDNN.py'

### Reading the command line
from argparse import ArgumentParser
import sys
from colorama import init, Fore
init(autoreset = True)

def ReadArgParser():
    parser = ArgumentParser()
    parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
    parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
    parser.add_argument('-s', '--Signal', help = 'Signal: \'VBFHVTWZ\', \'Radion\', \'RSG\' or \'VBFRadion\'', type = str, default = 'all')
    parser.add_argument('-j', '--JetCollection', help = 'Jet collection: \'TCC\'', type = str, default = 'TCC')
    parser.add_argument('-b', '--Background', help = 'Background: \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\' (in quotation mark separated by a space)', type = str, default = 'all')
    parser.add_argument('-t', '--TrainingFraction', help = 'Relative size of the training sample, between 0 and 1', default = 0.8)
    parser.add_argument('-p', '--PreselectionCuts', help = 'Preselection cut', type = str)
    parser.add_argument('-n', '--Nodes', help = 'Number of nodes of the (p)DNN, should always be >= nColumns and strictly positive', default = 32)
    parser.add_argument('-l', '--Layers', help = 'Number of hidden layers of the (p)DNN', default = 2)
    parser.add_argument('-e', '--Epochs', help = 'Number of epochs for the training', default = 150)
    parser.add_argument('-v', '--Validation', help = 'Fraction of the training data that will actually be used for validation', default = 0.2)
    parser.add_argument('-d', '--Dropout', help = 'Fraction of the neurons to drop during the training', default = 0.2)
    parser.add_argument('-m', '--Mass', help = 'Masses for the (P)DNN train/test (GeV, in quotation mark separated by a space)', default = 'all')
    
    args = parser.parse_args()

    analysis = args.Analysis
    if args.Analysis is None and sys.argv[0] != fileName1:
        parser.error(Fore.RED + 'Requested type of analysis (either \'mergered\' or \'resolved\')')
    elif args.Analysis != 'resolved' and args.Analysis != 'merged' and sys.argv[0] != fileName1:
        parser.error(Fore.RED + 'Analysis can be either \'merged\' or \'resolved\'')
    channel = args.Channel
    if args.Channel is None and sys.argv[0] != fileName1:
        parser.error(Fore.RED + 'Requested channel (either \'ggF\' or \'VBF\')')
    elif args.Channel != 'ggF' and args.Channel != 'VBF' and sys.argv[0] != fileName1:
        parser.error(Fore.RED + 'Channel can be either \'ggF\' or \'VBF\'')
    signal = args.Signal.split()
    signalString = 'all'
    if args.Signal != 'all':
        signalString = '_'.join([str(item) for item in signal])
    jetCollection = args.JetCollection
    if args.JetCollection is None:
        parser.error(Fore.RED + 'Requested jet collection (\'TCC\' or )')
    #elif args.JetCollection != 'TCC':
        #parser.error(Fore.RED + 'Jet collection can be \'TCC\', ')
    background = args.Background.split()
    for bkg in background:
        if (bkg !=  'Zjets' and bkg != 'Wjets' and bkg != 'stop' and bkg != 'Diboson' and bkg != 'ttbar' and bkg != 'all'):
            parser.error(Fore.RED + 'Background can be \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\'')
    backgroundString = 'all'
    if args.Background != 'all':
        backgroundString = '_'.join([str(item) for item in background])
    trainingFraction = float(args.TrainingFraction)
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
    mass = args.Mass.split()

    if sys.argv[0] == fileName1:
        return jetCollection

    if sys.argv[0] == fileName2:
        return jetCollection, analysis, channel, preselectionCuts

    if sys.argv[0] == fileName3:
        print(Fore.BLUE + '         background = ' + str(backgroundString))
        print(Fore.BLUE + '  training fraction = ' + str(trainingFraction))
        return jetCollection, analysis, channel, preselectionCuts, backgroundString, signalString, trainingFraction

    if(sys.argv[0] == fileName4 or sys.argv[0] == fileName5):
        print(Fore.BLUE + '          background(s) = ' + str(backgroundString))
        print(Fore.BLUE + '          test mass(es) = ' + str(mass))
        print(Fore.BLUE + '      training fraction = ' + str(trainingFraction))
        print(Fore.BLUE + '        number of nodes = ' + str(numberOfNodes))
        print(Fore.BLUE + 'number of hidden layers = ' + str(numberOfLayers))
        print(Fore.BLUE + '       number of epochs = ' + str(numberOfEpochs))
        print(Fore.BLUE + '    validation fraction = ' + str(validationFraction))
        print(Fore.BLUE + '                dropout = ' + str(dropout))
        return jetCollection, analysis, channel, preselectionCuts, backgroundString, trainingFraction, signalString, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, mass

### Reading from the configuration file
import configparser, ast

configurationFile = 'Configuration.ini'

def ReadConfigSaveToPkl(jetCollection):
    config = configparser.ConfigParser()
    config.read(configurationFile)
    ntuplePath = config.get('config', 'ntuplePath')
    inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
    dfPath = config.get('config', 'dfPath')
    dfPath += jetCollection + '/'
    print (format('Output directory: ' + Fore.GREEN + dfPath), checkCreateDir(dfPath))
    return ntuplePath, inputFiles, dfPath

def ReadConfig(analysis, jetCollection):
    config = configparser.ConfigParser()
    config.read(configurationFile)
    inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
    dataType = ast.literal_eval(config.get('config', 'dataType'))
    rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
    signalsList = ast.literal_eval(config.get('config', 'signals'))
    backgroundsList = ast.literal_eval(config.get('config', 'backgrounds'))
    dfPath = config.get('config', 'dfPath')
    dfPath += jetCollection + '/'# + '_DataFrames/'
    #modelPath = config.get('config', 'modelPath')
    #modelPath += jetCollection + '/'
    if analysis == 'merged':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesMerged'))
    elif analysis == 'resolved':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolved'))

    if sys.argv[0] == fileName2:
        return inputFiles, dataType, rootBranchSubSample, dfPath, InputFeatures
    if sys.argv[0] == fileName3:
        return dfPath, InputFeatures, signalsList, backgroundsList
    if sys.argv[0] == fileName4 or sys.argv[0] == fileName5:
        return dfPath, InputFeatures

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
import numpy as np
def LoadData(dfPath, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures):
    fileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + str(signal) + '_' + preselectionCuts + '_' + background + '_' + str(trainingFraction) + 't'
    data_Train = pd.read_pickle(dfPath + '/data_train_' + fileCommonName + '.pkl')
    data_Test = pd.read_pickle(dfPath + '/data_test_' + fileCommonName + '.pkl')
    data_Train_unscaled = pd.read_pickle(dfPath + '/data_train_unscaled_' + fileCommonName + '.pkl')
    X_Train_unscaled =  data_Train_unscaled[InputFeatures]
    m_Test_unscaled = pd.read_pickle(dfPath + '/m_test_unscaled_' + fileCommonName + '.pkl').values
    m_Train_unscaled = data_Train_unscaled['mass'].values
    return data_Train, data_Test, X_Train_unscaled, m_Train_unscaled, m_Test_unscaled

### Writing in the log file
def WritingLogFile(dfPath, X_test, y_test, X_train, y_train, InputFeatures, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, useWeights):
    logString = 'dfPath: ' + dfPath + '\nNumber of test events: ' + str(int(X_test.shape[0])) + ' (' + str(sum(y_test)) + ' signal and ' + str(len(y_test) - sum(y_test)) + ' background)' + '\nNumber of train events: ' + str(X_train.shape[0]) + ' (' + str(sum(y_train)) + ' signal and ' + str(len(y_train) - sum(y_train)) + ' background)' + '\nInputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nuseWeights: ' + str(useWeights)
    return logString

def WriteLogFile(numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, InputFeatures, dfPath):
    logString = 'Number of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nInputFeatures: ' + str(InputFeatures) + '\ndfPath: ' + dfPath# + '\nNumber of train events: ' + str(len(data_train)) + ' (' + str(len(data_train_signal)) + ' signal and ' + str(len(data_train_bkg)) + ' background)' + '\nNumber of test events: ' + str(len(data_test)) + ' (' + str(len(data_test_signal)) + ' signal and ' + str(len(data_test_bkg)) + ' background)'
    return logString

### Shuffling dataframe
import sklearn.utils

def ShufflingData(dataFrame):
    dataFrame = sklearn.utils.shuffle(dataFrame, random_state = 123)
    #dataFrame = dataFrame.reset_index(drop = True)
    return dataFrame

### Building the (P)DNN
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation

def BuildDNN(N_input, nodesNumber, layersNumber, dropout):
    model = Sequential()
    model.add(Dense(units = nodesNumber, input_dim = N_input))
    model.add(Activation('relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(0, layersNumber):
        model.add(Dense(nodesNumber))
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
    print(Fore.GREEN + 'Saved architecture in ' + outputArch)
    outputWeights = outputDir + '/weights.h5'
    model.save_weights(outputWeights)
    print(Fore.GREEN + 'Saved weights in ' + outputWeights)

def SaveVariables(outputDir, X_input):
    outputVar = outputDir + '/variables.json'
    with open(outputVar, 'w') as var_file:
        var_file.write("{\n")
        var_file.write("  \"inputs\": [\n")
        for col in X_input.columns:
            offset = -1. * float(X_input.mean(axis = 0)[col])
            scale = 1. / float(X_input.std(axis = 0)[col])
            var_file.write("    {\n")
            var_file.write("      \"name\": \"%s\",\n" % col)
            var_file.write("      \"offset\": %lf,\n" % offset) # EJS 2021-05-27: I have compelling reasons to believe this should be -mu
            var_file.write("      \"scale\": %lf\n" % scale) # EJS 2021-05-27: I have compelling reasons to believe this should be 1/sigma
            var_file.write("    }")
            '''
            if (col < X_input.shape[1]-1):
                var_file.write(",\n")
            else:
                var_file.write("\n")
            '''
            if col != X_input.columns[len(X_input.columns) - 1]:
                var_file.write(",\n")
            else:
                var_file.write("\n")
        var_file.write("  ],\n")
        var_file.write("  \"class_labels\": [\"BinaryClassificationOutputName\"]\n")
        var_file.write("}\n")
    print(Fore.GREEN + 'Saved variables in ' + outputVar)

def SaveFeatureScaling(outputDir, X_input):
    outputFeatureScaling = outputDir + '/FeatureScaling.dat'
    with open(outputFeatureScaling, 'w') as scaling_file: # EJS 2021-05-27: check which file name is hardcoded in the CxAODReader
        scaling_file.write("[")
        scaling_file.write(', '.join(str(i) for i in X_input.columns))
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
    print(Fore.GREEN + 'Saved features scaling in ' + outputFeatureScaling)
    
def SaveModel(model, X_input, outputDir):
    SaveArchAndWeights(model, outputDir)
    SaveVariables(outputDir, X_input)
    SaveFeatureScaling(outputDir, X_input)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams.update({'font.size': 16})

### Evaluating the (P)DNN performance
def EvaluatePerformance(model, X_test, y_test):
    perf = model.evaluate(X_test, y_test, batch_size = 2048)
    testLoss = perf[0]
    testAccuracy = perf[1]
    return testLoss, testAccuracy

### Prediction on signal and background separately
def PredictionSigBkg(model, X_train_signal, X_train_bkg, X_test_signal, X_test_bkg):
    yhat_train_signal = model.predict(X_train_signal, batch_size = 2048)
    yhat_train_bkg = model.predict(X_train_bkg, batch_size = 2048)
    yhat_test_signal = model.predict(X_test_signal, batch_size = 2048)
    yhat_test_bkg = model.predict(X_test_bkg, batch_size = 2048)
    return yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg

### Drawing Accuracy
def DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, useWeights, cutTrainEvents, mass = 0):
    plt.plot(modelMetricsHistory.history['accuracy'], label = 'Training')
    lines = plt.plot(modelMetricsHistory.history['val_accuracy'], label = 'Validation')
    xvalues = lines[0].get_xdata()
    #print(yvalues[len(yvalues) - 1])    
    plt.scatter([xvalues[len(xvalues) - 1]], [testAccuracy], label = 'Test', color = 'green')
    emptyPlot, = plt.plot([0, 0], [1, 1], color = 'white')
    titleAccuracy = NN + ' model accuracy'
    if NN == 'DNN':
        if mass >= 1000:
            titleAccuracy += ' (mass: ' + str(int(mass / 1000)) + ' TeV)'
        else:
            titleAccuracy += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    #plt.legend()
    #legend1 = plt.legend(['Training', 'Validation'], loc = 'lower right')
    legend1 = plt.legend(loc = 'center right')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg) + '\nuseWeights: ' + str(useWeights) + '\ncutTrainEvents: ' + str(cutTrainEvents)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    #legendText += '\nTest accuracy: ' + str(round(testAccuracy, 2))
    #plt.figtext(0.5, 0.3, legendText, wrap = True, horizontalalignment = 'left')
    #plt.legend(legendText)
    legend2 = plt.legend([emptyPlot], [legendText], frameon = False)
    plt.gca().add_artist(legend1)
    #plt.figtext(0.69, 0.28, 'Test accuracy: ' + str(round(testAccuracy, 2)), wrap = True, horizontalalignment = 'left')#, fontsize = 10)
    AccuracyPltName = outputDir + '/Accuracy.png'
    plt.savefig(AccuracyPltName)
    print(Fore.GREEN + 'Saved ' + AccuracyPltName)
    plt.clf()
        
### Drawing Loss
def DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, useWeights, cutTrainEvents, mass = 0):
    plt.plot(modelMetricsHistory.history['loss'], label = 'Training')
    lines = plt.plot(modelMetricsHistory.history['val_loss'], label = 'Validation')
    xvalues = lines[0].get_xdata()
    #print(yvalues[len(yvalues) - 1])    
    plt.scatter([xvalues[len(xvalues) - 1]], [testLoss], label = 'Test', color = 'green')
    #emptyPlot, = plt.plot([0, 0], [1, 1], color = 'white')
    titleLoss = NN + ' model loss'
    if NN == 'DNN':
        if mass >= 1000:
            titleLoss += ' (mass: ' + str(int(mass / 1000)) + ' TeV)'
        else:
            titleLoss += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleLoss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Training', 'Validation'], loc = 'upper right')
    legend1 = plt.legend(loc = 'upper right')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\npreselection cuts: ' + PreselectionCuts + '\nsignal: ' + signal + '\nbackground: ' + str(bkg) + '\nuseWeights: ' + str(useWeights) + '\ncutTrainEvents: ' + str(cutTrainEvents)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    #legendText += '\nTest loss: ' + str(round(testLoss, 2))
    plt.figtext(0.4, 0.4, legendText, wrap = True, horizontalalignment = 'left')
    #plt.figtext(0.7, 0.7, 'Test loss: ' + str(round(testLoss, 2)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    #legend2 = plt.legend([emptyPlot], [legendText], frameon = False, loc = 'center right')
    #plt.gca().add_artist(legend1)
    LossPltName = outputDir + '/Loss.png'
    plt.savefig(LossPltName)
    print(Fore.GREEN + 'Saved ' + LossPltName)
    plt.clf()
'''
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
def DrawROC(fpr, tpr, AUC, outputDir, mass):
    plt.plot(fpr,  tpr, color = 'darkorange', lw = 2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    titleROC = 'ROC curves (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleROC)
    plt.figtext(0.7, 0.25, 'AUC: ' + str(round(AUC, 2)), wrap = True, horizontalalignment = 'center')
    ROCPltName = outputDir + '/oldROC.png'
    plt.savefig(ROCPltName)
    print('Saved ' + ROCPltName)
    plt.clf()
''' 
def integral(y,x,bins):
    x_min=x
    s=0
    for i in np.where(bins>x)[0][:-1]:
        s=s+y[i]*(bins[i+1]-bins[i])
    return s

### Drawing scores, ROC and efficiency
#import numpy as np

def DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, savePlot, useWeights, cutTrainEvents):

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
        if mass >= 1000:
            titleScores = NN + ' scores (mass: ' + str(int(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            titleScores = NN + ' scores (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(titleScores)
        plt.legend(loc = 'upper center')
        legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg) + '\nuseWeights: ' + str(useWeights) + '\ncutTrainEvents: ' + str(cutTrainEvents)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        #plt.figtext(0.35, 0.45, legendText, wrap = True, horizontalalignment = 'left')
        ScoresPltName = outputDir + '/Scores_' + bkg + '.png'
        plt.savefig(ScoresPltName)
        print(Fore.GREEN + 'Saved ' + ScoresPltName)
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
        if mass >= 1000:
            titleROC = NN + ' ROC curve (mass: ' + str(int(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            titleROC = NN + ' ROC curve (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(titleROC)
        legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg) + '\nuseWeights: ' + str(useWeights) + '\ncutTrainEvents: ' + str(cutTrainEvents)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        plt.figtext(0.4, 0.25, legendText, wrap = True, horizontalalignment = 'left')
        plt.figtext(0.4, 0.2, 'AUC: ' + str(round(Area, 2)), wrap = True, horizontalalignment = 'center')
        ROCPltName = outputDir + '/ROC_' + bkg + '.png'
        plt.savefig(ROCPltName)
        print(Fore.GREEN + 'Saved ' + ROCPltName)
        plt.clf()

    ### Background rejection vs efficiency
    WP=[0.90,0.94,0.97,0.99]
    rej=1./bkg_eff
    WP_idx=[np.where(np.abs(signal_eff-WP[i])==np.min(np.abs(signal_eff-WP[i])))[0][0] for i in range(0,len(WP))]
    WP_rej=[str(round(10*rej[WP_idx[i]])/10) for i in range(0,len(WP))]
    print(Fore.BLUE + 'Working points (WP): ' + str(WP))
    #print(Fore.BLUE + 'Working points (WP): ' + str(bins_0[Nbins-np.array(WP_idx)]))
    print(Fore.BLUE + 'Background rejection at each WP: ' + str(WP_rej))

    if savePlot:
        plt.plot(signal_eff,rej)
        for i in range(0,len(WP)):
            plt.axvline(x=WP[i],color='Red',linestyle='dashed',label='Bkg Rejection @ '+str(WP[i])+' WP: '+WP_rej[i])
        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection')
        plt.xlim([0.85,1])
        plt.yscale('log')
        if mass >= 1000:
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(int(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(BkgRejTitle)
        plt.legend()
        EffPltName = outputDir + '/BkgRejection_' + bkg +'.png'
        plt.savefig(EffPltName)
        print(Fore.GREEN + 'Saved ' + EffPltName)
        plt.clf()
    return Area, WP, WP_rej 

from sklearn.metrics import confusion_matrix
import itertools

def DrawCM(yhat_test, y_test, normalize, outputDir, mass, background):
    yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test])
    cm = confusion_matrix(y_test, yResult_test_cls)
    classes = ['Background', 'Signal']
    np.set_printoptions(precision = 2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    cmap = plt.cm.Oranges
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    if mass >= 1000:
        titleCM = 'Confusion matrix (mass: ' + str(int(mass / 1000)) + ' TeV, bkg: ' + background + ')'
    else:
        titleCM = 'Confusion matrix (mass: ' + str(int(mass)) + ' GeV, bkg: ' + background + ')'
    plt.title(titleCM)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation = 90)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    CMPltName = outputDir + '/ConfusionMatrix_' + background + '.png'
    plt.savefig(CMPltName)
    print(Fore.GREEN + 'Saved ' + CMPltName)
    plt.clf()    

def weightEvents(origin_train):
    originsList = np.array(list(set(list(origin_train))))
    originsNumber = np.array([])
    for origin in originsList:
        originsNumber = np.append(originsNumber, list(origin_train).count(origin))
    minNumber = min(originsNumber)
    weights = minNumber / originsNumber
    w_origin_train = origin_train.copy()
    for origin in originsList:
        w_origin_train = np.where(w_origin_train == str(origin), weights[np.where(originsList == origin)], w_origin_train)
    w_origin_train = np.asarray(w_origin_train).astype(np.float32)
    return w_origin_train

def cutEventsOld(X_train_mass, origin_train_mass):
    X_train_mass_ext = np.insert(X_train_mass, X_train_mass.shape[1], origin_train_mass, axis = 1)
    X_train_signal_mass_ext = X_train_mass_ext[origin_train_mass == 0]
    X_train_signal_mass_ext = np.insert(X_train_signal_mass_ext, X_train_signal_mass_ext.shape[1], 1, axis = 1)
    '''
    X_train_bkg_mass_ext = X_train_mass_ext[origin_train_mass != 0]
    X_train_bkg_mass_ext = np.insert(X_train_bkg_mass_ext, X_train_bkg_mass_ext.shape[1], 0, axis = 1)
    '''
    originsList = list(set(list(origin_train_mass)))
    originsNumber = np.array([])
    for origin in originsList:
        originsNumber = np.append(originsNumber, list(origin_train_mass).count(origin))
    print(originsNumber)
    minNumber = min(originsNumber)
    print(minNumber)
    X_train_mass_origin_ext = X_train_signal_mass_ext
    for origin in range(1, 3):
        print('concat')
        X_train_origin_ext = X_train_mass_ext[origin_train_mass == origin]
        X_train_origin_ext = X_train_origin_ext[:int(minNumber)]
        X_train_origin_ext = np.insert(X_train_origin_ext, X_train_origin_ext.shape[1], 0, axis = 1)
        X_train_mass_origin_ext = np.concatenate((X_train_mass_origin_ext, X_train_origin_ext), axis = 0)
    '''
    origin_train_mass = X_train_mass_origin_ext[:, X_train_mass_origin_ext.shape[1] - 1]
    for origin in origin_train_mass:
        if origin == 0:
            X_train_mass_origin_ext = np.insert(X_train_mass_origin_ext, X_train_mass_origin_ext.shape[1], 1, axis = 1)
        else:
            X_train_mass_origin_ext = np.insert(X_train_mass_origin_ext, X_train_mass_origin_ext.shape[1], 0, axis = 1)
    '''
    X_train_mass_origin_ext = ShufflingData(X_train_mass_origin_ext)
    print(X_train_mass_origin_ext.shape[0])
    y_train_mass_origin = X_train_mass_origin_ext[:, X_train_mass_origin_ext.shape[1] - 1]
    X_train_mass_origin_ext = np.delete(X_train_mass_origin_ext, X_train_mass_origin_ext.shape[1] - 1, axis = 1)
    origin_train_mass = X_train_mass_origin_ext[:, X_train_mass_origin_ext.shape[1] - 1]
    X_train_bkg_origin_ext = X_train_mass_origin_ext[origin_train_mass != 0]
    origin_train_bkg_mass = X_train_bkg_origin_ext[:, X_train_bkg_origin_ext.shape[1] - 1]
    X_train_mass_origin = np.delete(X_train_mass_origin_ext, X_train_mass_origin_ext.shape[1] - 1, axis = 1)
    return X_train_mass_origin, y_train_mass_origin, origin_train_bkg_mass    

def cutEvents(data_train_mass):
    data_train_mass_cut = pd.DataFrame()
    originsList = np.array(list(set(list(data_train_mass['origin']))))
    originsNumber = np.array([])
    for origin in originsList:
        originsNumber = np.append(originsNumber, list(data_train_mass['origin']).count(origin))
    minNumber = int(min(originsNumber))
    for origin in originsList:
        data_train_mass_origin = data_train_mass[data_train_mass['origin'] == origin]
        if origin == 'Radion':
            data_train_mass_origin = data_train_mass_origin
        else:
            data_train_mass_origin = data_train_mass_origin[:int(minNumber)]
        data_train_mass_cut = pd.concat([data_train_mass_cut, data_train_mass_origin], ignore_index = True)
    return data_train_mass_cut
