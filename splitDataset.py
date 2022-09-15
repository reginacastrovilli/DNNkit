### This script takes as input the dataframes produced in the previous step (buildDataset), computes train weights to compensate for different
### statistics for each signal mass value and for signal/background, scales each feature according to median and interquartile range 
### and splits the resulting dataframe in train and test samples.
### Histograms of scaled and unscaled variables will be saved if running with --drawPlots = 1

import seaborn
from Functions import ReadArgParser, checkCreateDir, ReadConfig, DrawVariablesHisto, ShufflingData, ComputeTrainWeights, ScalingFeatures#, SaveFeatureScaling,
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import shlex
from colorama import init, Fore
init(autoreset = True)

#print(sys.executable, ' '.join(map(shlex.quote, sys.argv)))
#from tensorflow.keras.utils import to_categorical

pd.options.mode.chained_assignment = None ### to suppress the SettingWithCopyWarning

### Reading from command line
tag, jetCollection, analysis, channel, preselectionCuts, background, signal, trainingFraction, drawPlots = ReadArgParser()

### Reading from configuration file
dfPath, InputFeatures, signalsList, backgroundsList = ReadConfig(tag, analysis, jetCollection, signal)

### Loading input file
inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background
fileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
data = pd.read_pickle(inputDir + '/MixData_' + fileCommonName + '.pkl') 

### If not already existing, creating output directory
outputDir = inputDir
checkCreateDir(outputDir)
fileCommonName += '_' + str(trainingFraction) + 't'

### Creating log file
logFileName = '/logFile_splitDataset_' + fileCommonName + '.txt'
logFile = open(outputDir + logFileName, 'w')
logFile.write('Command executed: ' + sys.executable + ' '.join(map(shlex.quote, sys.argv)) + '\nTag: ' + tag + '\nJet collection: ' + jetCollection + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + preselectionCuts + '\nSignal: ' + signal + '\nBackground: ' + background)

### Creating the list of backgrounds and signal processes to select
if background == 'all':
    inputOrigins = backgroundsList.copy()
    backgroundLegend = backgroundsList.copy()
    logFile.write(' (' + str(backgroundsList) + ')')
else:
    inputOrigins = list(background.split('_'))
inputOrigins.append(signal)
    
### Selecting events according to their origin 
data_set = data[data['origin'].isin(inputOrigins)]

### Dividing signal from background
dataSetSignal = data_set[data_set['origin'] == signal]
dataSetBackground = data_set[data_set['origin'] != signal]

massesSignalList = sorted(list(set(list(dataSetSignal['mass']))))
print(Fore.BLUE + 'Masses in the signal sample: ' + str(massesSignalList) + ' GeV (' + str(len(massesSignalList)) + ')')
logFile.write('\nMasses in the signal sample: ' + str(massesSignalList) + ' GeV (' + str(len(massesSignalList)) + ')')

### Creating new column in the dataframes with train weight
dataSetSignal, dataSetBackground, logString = ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts, drawPlots)
if logString != '':
    print(Fore.GREEN + logString)
    logFile.write(logString)

### Printing and saving information on events numbers 
stringForBkgEvents = 'Number of background events: ' + str(dataSetBackground.shape[0]) + ' raw, ' + str(dataSetBackground['weight'].sum()) + ' with MC weights, ' + str(dataSetBackground['train_weight'].sum()) + ' with train weights'
print(Fore.BLUE + stringForBkgEvents)
logFile.write('\n' + stringForBkgEvents)
if(len(backgroundsList) > 1):
    for bkg in backgroundsList:
        dataSetSingleBackground = dataSetBackground[dataSetBackground['origin'] == bkg]
        stringToSaveBkg = '   ---> Number of ' + bkg + ' events: ' + str(dataSetSingleBackground.shape[0]) + ' raw, ' + str(dataSetSingleBackground['weight'].sum()) + ' with MC weights, ' + str(dataSetSingleBackground['train_weight'].sum()) + ' with train weights'
        print(Fore.BLUE + stringToSaveBkg)
        logFile.write('\n' + stringToSaveBkg)
stringForSignalEvents = 'Number of signal events: ' + str(dataSetSignal.shape[0]) + ' raw, ' + str(dataSetSignal['weight'].sum()) + ' with MC weights, ' + str(dataSetSignal['train_weight'].sum()) + ' with train weights'
print(Fore.BLUE + stringForSignalEvents)
logFile.write('\n' + stringForSignalEvents)
for signalMass in massesSignalList:
    dataSetSignalMass = dataSetSignal[dataSetSignal['mass'] == signalMass]
    stringToSaveSignal = '   ---> Number of signal events with mass ' + str(signalMass) + ' GeV: ' + str(dataSetSignalMass.shape[0]) + ' raw, ' + str(dataSetSignalMass['weight'].sum()) + ' with MC weights, ' + str(dataSetSignalMass['train_weight'].sum()) + ' with train weights'
    print(Fore.BLUE + stringToSaveSignal)
    logFile.write('\n' + stringToSaveSignal)

### Concatening signal and background dataframes
dataFrame = pd.concat([dataSetSignal, dataSetBackground], ignore_index = True)

### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Creating a new column in the dataFrame that will store the unscaled mass
dataFrame = dataFrame.assign(unscaledMass = dataFrame['mass'])

### Splitting data into train and test set
data_train, data_test = train_test_split(dataFrame, train_size = trainingFraction)

'''
### Slicing data train for statistic test
nEvents = int(data_train.shape[0] / 4)
data_train = data_train[:nEvents]
'''

### Scaling InputFeatures of train and test set
data_train, data_test, logString = ScalingFeatures(data_train, data_test, InputFeatures, outputDir)
logFile.write(logString)

if drawPlots:
    scaledHistogramsPath = outputDir + '/trainScaledHistograms'
    print(format('Output directory: ' + Fore.GREEN + scaledHistogramsPath), checkCreateDir(scaledHistogramsPath))
    logFile.write('\nSaving histograms of the scaled train dataset in ' + scaledHistogramsPath)
    DrawVariablesHisto(data_train, InputFeatures, scaledHistogramsPath, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts, True)

### Saving scaled dataframes
dataTrainName = outputDir + '/data_train_' + fileCommonName + '.pkl'
data_train.to_pickle(dataTrainName)
print(Fore.GREEN + 'Saved ' + dataTrainName)
logFile.write('\nSaved scaled train dataframe in ' + dataTrainName)
dataTestName = outputDir + '/data_test_' + fileCommonName + '.pkl'
data_test.to_pickle(dataTestName)
print(Fore.GREEN + 'Saved ' + dataTestName)
logFile.write('\nSaved scaled test dataframe in ' + dataTestName)

### Closing log file
logFile.close()
print(Fore.GREEN + 'Saved ' + outputDir + logFileName)
