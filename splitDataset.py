### This script takes as input the dataframes produced in the previous step (buildDataset), computes train weights to compensate for different
### statistics for each signal mass value and for signal/background, scales each feature according to median and interquartile range 
### and splits the resulting dataframe in train and test samples.
### Histograms of scaled and unscaled variables can be saved

from Functions import ReadArgParser, checkCreateDir, ReadConfig, DrawVariablesHisto, ShufflingData, ComputeTrainWeights, ScalingFeatures#, SaveFeatureScaling,
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from colorama import init, Fore
init(autoreset = True)
#from tensorflow.keras.utils import to_categorical
pd.options.mode.chained_assignment = None ### to suppress the SettingWithCopyWarning

### Reading from command line
tag, jetCollection, analysis, channel, preselectionCuts, background, signal, trainingFraction, drawPlots = ReadArgParser()

### Reading from configuration file
dfPath, InputFeatures, signalsList, backgroundsList = ReadConfig(tag, analysis, jetCollection)

### Loading input file
inputDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background
fileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
data = pd.read_pickle(inputDir + '/MixData_' + fileCommonName + '.pkl') 

### If not already existing, creating output directory
outputDir = inputDir + '_fullStat'
checkCreateDir(outputDir) 
fileCommonName += '_' + str(trainingFraction) + 't'

### Creating log file
logFileName = '/logFileSplitDataset_' + fileCommonName + '.txt'
logFile = open(outputDir + logFileName, 'w')
logFile.write('Tag: ' + tag + '\nJet collection: ' + jetCollection + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + preselectionCuts + '\nSignal: ' + signal + '\nBackground: ' + background)

### Creating the list of backgrounds and signal processes to select
if background == 'all':
    inputOrigin = backgroundsList.copy()
    backgroundLegend = backgroundsList.copy()
    logFile.write(' (' + str(backgroundsList) + ')')
else:
    inputOrigin = list(background.split('_'))
inputOrigin.append(signal)
    
### Selecting events according to their origin 
data_set = data[data['origin'].isin(inputOrigin)]

dictOrigin = {}
for origin in inputOrigin:
    dictOrigin[origin] = list(data_set['origin']).count(origin)
    print(Fore.BLUE + 'Number of ' + origin + ' events: ' + str(dictOrigin[origin]))
    logFile.write('\nNumber of ' + origin + ' events: ' + str(dictOrigin[origin]))
        
### Dividing signal from background
dataSetSignal = data_set[data_set['origin'] == signal]
dataSetBackground = data_set[data_set['origin'] != signal]
print(Fore.BLUE + '---> Number of background events: ' + str(dataSetBackground.shape[0]))
logFile.write('\n---> Number of background events: ' + str(dataSetBackground.shape[0]))
print(Fore.BLUE + '---> Number of signal events: ' + str(dataSetSignal.shape[0]))
logFile.write('\n---> Number of signal events: ' + str(dataSetSignal.shape[0]))
massesSignalList = sorted(list(set(list(dataSetSignal['mass']))))
print(Fore.BLUE + 'Masses in the signal sample: ' + str(massesSignalList) + ' GeV (' + str(len(massesSignalList)) + ')')
logFile.write('\nMasses in the signal sample: ' + str(massesSignalList) + ' GeV (' + str(len(massesSignalList)) + ')')

### Creating new column in the dataframes with train weight
dataSetSignal, dataSetBackground, logString = ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts, drawPlots)
logFile.write(logString)

### Concatening signal and background dataframes
dataFrame = pd.concat([dataSetSignal, dataSetBackground], ignore_index = True)

### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Creating a new column in the dataFrame that will store the unscaled mass
dataFrame = dataFrame.assign(unscaledMass = dataFrame['mass'])

### Splitting data into train and test set
data_train, data_test = train_test_split(dataFrame, train_size = trainingFraction)

if drawPlots:
    trainHistogramsPath = outputDir + '/trainUnscaledHistograms'
    print(format('Output directory: ' + Fore.GREEN + trainHistogramsPath), checkCreateDir(trainHistogramsPath))
    logFile.write('\nSaving histograms of the unscaled train dataset in ' + trainHistogramsPath)
    DrawVariablesHisto(data_train, InputFeatures, trainHistogramsPath, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts)

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
    logFile.write('\nSaving histograms of the unscaled train dataset in ' + scaledHistogramsPath)
    DrawVariablesHisto(data_train, InputFeatures, scaledHistogramsPath, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts)

### Saving scaled dataframes
dataTrainName = outputDir + '/data_train_' + fileCommonName + '.pkl'
data_train.to_pickle(dataTrainName)
print(Fore.GREEN + 'Saved ' + dataTrainName)
logFile.write('\nSaved scaled train dataframe in ' + dataTrainName)
dataTestName = outputDir + '/data_test_' + fileCommonName + '.pkl'
data_test.to_pickle(dataTestName)
print(Fore.GREEN + 'Saved ' + dataTestName)
logFile.write('\nSaved scaled test dataframe in ' + dataTestName)

logFile.close()
print(Fore.GREEN + 'Saved ' + outputDir + logFileName)
