from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectEvents, CutMasses
import pandas as pd
import numpy as np
import re
import ast
import random
import os.path
from colorama import init, Fore
init(autoreset = True)

### Reading the command line
jetCollection, analysis, channel, preselectionCuts, signal, background = ReadArgParser()

originsBkgTest = list(background.split('_'))

### Creating the list of the origins selected (signal + background)
targetOrigins = originsBkgTest.copy()
targetOrigins.insert(0, signal)

### Reading from config file
inputFiles, rootBranchSubSample, dfPath, InputFeatures = ReadConfig(analysis, jetCollection)

InputFeatures.append('isSignal')
InputFeatures.append('origin')

### Loading DSID-mass map and storing it into a dictionary
DictDSID = {}
DSIDfile = open('DSIDtoMass.txt')
lines = DSIDfile.readlines()
for line in lines:
    DictDSID[int(line.split(':')[0])] = int(line.split(':')[1])

dataFrameSignal = []
dataFrameBkg = []

for file in os.listdir(dfPath):
    for target in targetOrigins:
        if file.startswith(target):
            print(Fore.GREEN + 'Loading ' + dfPath + file)
            inputDf = pd.read_pickle(dfPath + file)
            inputDf = inputDf.assign(origin = target)
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

### Selecting events according to the type of analysis and channel
dataFrameSignal = SelectEvents(dataFrameSignal, channel, analysis, preselectionCuts)
dataFrameBkg = SelectEvents(dataFrameBkg, channel, analysis, preselectionCuts)

### Converting DSID to mass in the signal dataframe
massesSignal = dataFrameSignal['DSID'].copy()
DSIDsignal = np.array(list(set(list(dataFrameSignal['DSID']))))
for DSID in DSIDsignal:
    massesSignal = np.where(massesSignal == DSID, DictDSID[DSID], massesSignal)
dataFrameSignal = dataFrameSignal.assign(mass = massesSignal)

### Cutting signal events according to their mass and the type of analysis
dataFrameSignal = CutMasses(dataFrameSignal, analysis)
print(Fore.BLUE + 'Masses in the signal sample: ' + str(set(list(dataFrameSignal['mass']))))

### Assigning a random mass to background events according to the signal mass distribution 
massDict = dict(dataFrameSignal['mass'].value_counts(normalize = True))
massesBkg = random.choices(list(massDict.keys()), weights = list(massDict.values()), k = len(dataFrameBkg))
dataFrameBkg = dataFrameBkg.assign(mass = massesBkg)

### Concatening signal and background dataframes
dataFrame = pd.concat([dataFrameSignal, dataFrameBkg], ignore_index = True)

### Selecting in the dataframe only the variables that will be given as input to the neural networks
dataFrame = dataFrame[InputFeatures]

### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Saving pkl files
outputDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
outputFileName = '/MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + '.pkl'
dataFrame.to_pickle(outputDir + outputFileName)
print('Saved ' + outputDir + outputFileName)
