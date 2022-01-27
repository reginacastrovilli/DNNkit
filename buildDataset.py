from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectEvents, CutMasses, DrawVariablesHisto
import pandas as pd
import numpy as np
import re
import ast
import random
import os.path
from colorama import init, Fore
init(autoreset = True)
import matplotlib
import matplotlib.pyplot as plt

drawPlots = True

### Reading the command line
tag, jetCollection, analysis, channel, preselectionCuts, signal, background = ReadArgParser()

### Reading from config file
inputFiles, rootBranchSubSample, dfPath, variablesToSave, backgroundsList = ReadConfig(tag, analysis, jetCollection)

### Creating output directory and logFile
outputDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
fileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
logFileName = outputDir + '/EventsNumberAfterSelection_' + fileCommonName + '.txt'
logFile = open(logFileName, 'w')
logFile.write('CxAOD tag: ' + tag + '\nJetCollection: ' + jetCollection + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + '\nSignal: ' + signal + '\nBackground: ' + background + '\nInput files path: ' + dfPath + '\nInput files and number of events after selection:\n')

### Creating the list of the origins selected (signal + background)
if background == 'all':
    originsBkgTest = backgroundsList.copy()
else:
    originsBkgTest = list(background.split('_'))

targetOrigins = originsBkgTest.copy()
targetOrigins.insert(0, signal)

### Loading DSID-mass map and storing it into a dictionary
DictDSID = {}
DSIDfile = open('DSIDtoMass.txt')
lines = DSIDfile.readlines()
for line in lines:
    DictDSID[int(line.split(':')[0])] = int(line.split(':')[1])

### Creating two empty dataframes to fill with signal and background events
dataFrameSignal = []
dataFrameBkg = []

for file in os.listdir(dfPath):
    if not file.endswith('.pkl'):
        continue
    #if file == 'Wjets-mc16d_DF.pkl':
    #    continue
    for target in targetOrigins:
        if file.startswith(target):
            print(Fore.GREEN + 'Loading ' + dfPath + file)
            inputDf = pd.read_pickle(dfPath + file)
            logFile.write('\n' + file)
            ### Selecting only variables relevant to the analysis (needed in order to avoid memory issues)
            inputDf = inputDf[rootBranchSubSample]
            ### Selecting events according to merged/resolved regime and ggF/VBF channel
            inputDf = SelectEvents(inputDf, channel, analysis, preselectionCuts)
            logFile.write(' -> ' + str(inputDf.shape[0]))
            ### Creating new column in the dataframe with the origin
            inputDf = inputDf.assign(origin = target)
            ### Filling signal/background dataframes
            if target == signal:
                dataFrameSignal.append(inputDf)
            else:
                dataFrameBkg.append(inputDf)

### Concatening signal and background dataframes
dataFrameSignal = pd.concat(dataFrameSignal, ignore_index = True)
dataFrameBkg = pd.concat(dataFrameBkg, ignore_index = True)

### Creating a new isSignal column with values 1 (0) for signal (background) events
dataFrameSignal = dataFrameSignal.assign(isSignal = 1)
dataFrameBkg = dataFrameBkg.assign(isSignal = 0)

### Converting DSID to mass in the signal dataframe
massesSignal = dataFrameSignal['DSID'].copy()
DSIDsignal = np.array(list(set(list(dataFrameSignal['DSID']))))
for DSID in DSIDsignal:
    massesSignal = np.where(massesSignal == DSID, DictDSID[DSID], massesSignal)
dataFrameSignal = dataFrameSignal.assign(mass = massesSignal)

### Cutting signal events according to their mass and the type of analysis
dataFrameSignal = CutMasses(dataFrameSignal, analysis)
massesSignalList = list(set(list(dataFrameSignal['mass'])))
print(Fore.BLUE + 'Masses in the signal sample: ' + str(np.sort(np.array(massesSignalList))))
logFile.write('\nMasses in the signal sample: ' + str(np.sort(np.array(massesSignalList))))

'''
### Assigning a random mass to background events according to the signal mass distribution 
massDict = dict(dataFrameSignal['mass'].value_counts(normalize = True))
massesBkg = random.choices(list(massDict.keys()), weights = list(massDict.values()), k = len(dataFrameBkg))
dataFrameBkg = dataFrameBkg.assign(mass = massesBkg)
'''

### Assigning a random mass to background events 
massesBkg = np.random.choice(massesSignalList, dataFrameBkg.shape[0])
dataFrameBkg = dataFrameBkg.assign(mass = massesBkg)

### Concatening signal and background dataframes
dataFrame = pd.concat([dataFrameSignal, dataFrameBkg], ignore_index = True)

### Selecting in the dataframe only the variables relevant for the next steps
dataFrame = dataFrame[variablesToSave]

### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Saving pkl files
outputFileName = '/MixData_' + fileCommonName + '.pkl'
dataFrame.to_pickle(outputDir + outputFileName)
print(Fore.GREEN + 'Saved ' + outputDir + outputFileName)
logFile.write('\nSaved ' + outputDir + outputFileName)
logFile.close()
print(Fore.GREEN + 'Saved ' + logFileName)

### Drawing histogram of variables
if drawPlots:
    DrawVariablesHisto(dataFrame, outputDir, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts)
