### This script takes as input the pkl files resulting from the conversion of the ntuples produced by the Reader, 
### selects the events according to the regime (merged/resolved) and channel (ggF/VBF) requested, add some useful information to each event (origin, isSignal, mass)
### and saves the resulting dataframe. 
### A cut on the mass value is performed according to the regime selected.
### Histograms and a correlation matrix of the relevat variables can be saved.

from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectEvents, CutMasses, DrawVariablesHisto, DrawCorrelationMatrix
import pandas as pd
import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt
#import sys
#import shlex
from colorama import init, Fore
init(autoreset = True)
import math

#print(sys.executable, ' '.join(map(shlex.quote, sys.argv)))

overwriteDataFrame = False
#print(str(sys))
### Reading the command line
tag, jetCollection, analysis, channel, preselectionCuts, signal, background, drawPlots = ReadArgParser()

### Reading from config file
InputFeatures, dfPath, variablesToSave, backgroundsList = ReadConfig(tag, analysis, jetCollection, signal)

### Creating output directories and logFile
tmpOutputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts# + '/ggFandVBF'# + '/ggFVBF'
print(format('First output directory: ' + Fore.GREEN + tmpOutputDir), checkCreateDir(tmpOutputDir))
tmpFileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts

outputDir = tmpOutputDir + '/' + signal + '/' + background
print(format('Second output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
fileCommonName = tmpFileCommonName + '_' + signal + '_' + background
logFileName = outputDir + '/logFile_buildDataset_' + fileCommonName + '.txt'

logFile = open(logFileName, 'w')
logFile.write('Input files path: ' + dfPath + 'CxAOD tag: ' + tag + '\nJet collection: ' + jetCollection + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + preselectionCuts + '\nSignal: ' + signal + '\nBackground: ' + background)

### Loading DSID-mass map and storing it into a dictionary
DictDSID = {}
DSIDfileName = 'DSIDtoMass.txt'
with open(DSIDfileName) as DSIDfile:
    print(Fore.GREEN + 'Reading DSID - mass correspondance from ' + DSIDfileName)
    lines = DSIDfile.readlines()
    for line in lines:
        DictDSID[int(line.split(':')[0])] = int(line.split(':')[1])

### Creating the list of the origins selected (signal + background)
if background == 'all':
    inputOrigins = backgroundsList.copy()
    backgroundLegend = backgroundsList.copy()
    logFile.write(' (' + str(backgroundsList) + ')')
else:
    inputOrigins = list(background.split('_'))
inputOrigins.append(signal)
print(dfPath)

### Creating empty signal and background dataframe 
dataFrameSignal = []
dataFrameBkg = []
'''
def computePx(ptArray, phi):
    px = ptArray * np.cos(phi)
    return px

def computePy(ptArray, phi):
    px = ptArray * np.sin(phi)
    return px

def computePz(ptArray, eta):
    pz = ptArray * np.sinh(eta)
    return pz

def computeE(ptArray, etaArray, massArray):
    e = np.sqrt((ptArray ** 2) * (1 + np.sinh(etaArray) ** 2) + (massArray ** 2))
    return e
'''
for target in inputOrigins:
    fileName = tmpOutputDir + '/' + target + '_' + tmpFileCommonName + '.pkl'

    ### Loading dataframe if found and overwrite flag is false 
    if not overwriteDataFrame:
        if os.path.isfile(fileName):
            if target == signal:
                print(Fore.GREEN + 'Found signal dataframe: loading ' + fileName)
                dataFrameSignal = pd.read_pickle(fileName)
            else:
                print(Fore.GREEN + 'Found background dataframe: loading ' + fileName)
                dataFrameBkg.append(pd.read_pickle(fileName))

    ### Creating dataframe if not found or overwrite flag is true
    if overwriteDataFrame or not os.path.isfile(fileName):
        ### Defining local dataframe (we might have found only one among many dataframes)
        partialDataFrameBkg = []
        for file in os.listdir(dfPath):
            ### Loading input file
            if file.startswith(target) and file.endswith('.pkl'):
                '''
                if target != signal:
                    continue
                '''
                print(Fore.GREEN + 'Loading ' + dfPath + file)
                inputDf = pd.read_pickle(dfPath + file)
                ### Selecting events according to merged/resolved regime and ggF/VBF channel
                inputDf = SelectEvents(inputDf, channel, analysis, preselectionCuts, signal)
                ### Creating new column in the dataframe with the origin
                inputDf = inputDf.assign(origin = target)

                ### Filling signal/background dataframes
                if target == signal:
                    dataFrameSignal.append(inputDf)
                else:
                    partialDataFrameBkg.append(inputDf)
        ### Concatening and saving signal and background dataframes
        if target == signal:
            dataFrameSignal = pd.concat(dataFrameSignal, ignore_index = True)
            dataFrameSignal.to_pickle(fileName)

        elif target != signal:
            partialDataFrameBkg = pd.concat(partialDataFrameBkg, ignore_index = True)
            partialDataFrameBkg.to_pickle(fileName)
            ### Appending the local background dataframe to the final one
            dataFrameBkg.append(partialDataFrameBkg)

        print(Fore.GREEN + 'Saved ' + fileName)

### Concatening the global background dataframe
dataFrameBkg = pd.concat(dataFrameBkg, ignore_index = True)

### Creating a new isSignal column with values 1 (0) for signal (background) events
dataFrameSignal = dataFrameSignal.assign(isSignal = 1)
dataFrameBkg = dataFrameBkg.assign(isSignal = 0)
'''
### Removing isolated event with high lep1_m
dataFrameBkg = dataFrameBkg.query('lep1_m < 0.15')
'''
### Converting DSID to mass in the signal dataframe
massesSignal = dataFrameSignal['DSID'].copy()
DSIDsignal = np.array(list(set(list(dataFrameSignal['DSID']))))

for DSID in DSIDsignal:
    massesSignal = np.where(massesSignal == DSID, DictDSID[DSID], massesSignal)
dataFrameSignal = dataFrameSignal.assign(mass = massesSignal)

### Cutting signal events according to their mass and the type of analysis
dataFrameSignal = CutMasses(dataFrameSignal, analysis)
massesSignalList = list(set(list(dataFrameSignal['mass'])))
print(Fore.BLUE + 'Masses in the signal sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV (' + str(len(massesSignalList)) + ')')
logFile.write('\nMasses in the signal sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV (' + str(len(massesSignalList)) + ')')

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

'''
print(dataFrame)
cosphi = np.cos(dataFrame['lep1_phi'])
##################### 
lep1_px_array = computePx(dataFrame['lep1_pt'], dataFrame['lep1_phi'])
dataFrame = dataFrame.assign(lep1_px = lep1_px_array)
lep2_px_array = computePx(dataFrame['lep2_pt'], dataFrame['lep2_phi'])
dataFrame = dataFrame.assign(lep2_px = lep2_px_array)
fatjet_px_array = computePx(dataFrame['fatjet_pt'], dataFrame['fatjet_phi'])
dataFrame = dataFrame.assign(fatjet_px = fatjet_px_array)

lep1_py_array = computePy(dataFrame['lep1_pt'], dataFrame['lep1_phi'])
dataFrame = dataFrame.assign(lep1_py = lep1_py_array)
lep2_py_array = computePy(dataFrame['lep2_pt'], dataFrame['lep2_phi'])
dataFrame = dataFrame.assign(lep2_py = lep2_py_array)                
fatjet_py_array = computePy(dataFrame['fatjet_pt'], dataFrame['fatjet_phi'])
dataFrame = dataFrame.assign(fatjet_py = fatjet_py_array)                

lep1_pz_array = computePz(dataFrame['lep1_pt'], dataFrame['lep1_eta'])
dataFrame = dataFrame.assign(lep1_pz = lep1_pz_array)
lep2_pz_array = computePz(dataFrame['lep2_pt'], dataFrame['lep2_eta'])
dataFrame = dataFrame.assign(lep2_pz = lep2_pz_array)                
fatjet_pz_array = computePz(dataFrame['fatjet_pt'], dataFrame['fatjet_eta'])
dataFrame = dataFrame.assign(fatjet_pz = fatjet_pz_array)                

lep1_e_array = computeE(dataFrame['lep1_pt'], dataFrame['lep1_eta'], dataFrame['mass'])
dataFrame = dataFrame.assign(lep1_e = lep1_e_array)
lep2_e_array = computeE(dataFrame['lep2_pt'], dataFrame['lep2_eta'], dataFrame['mass'])
dataFrame = dataFrame.assign(lep2_e = lep2_e_array)
fatjet_e_array = computeE(dataFrame['fatjet_pt'], dataFrame['fatjet_eta'], dataFrame['mass'])
dataFrame = dataFrame.assign(fatjet_e = fatjet_e_array)
#####################
print(dataFrame)
#exit()
'''
'''
### Removing events with high absoulte MC weights
meanWeight = dataFrame['weight'].mean()
stdWeight = dataFrame['weight'].std()
selection = 'abs(weight - ' + str(meanWeight) + ') <= 5 * ' + str(stdWeight)
dataFrame = dataFrame.query(selection)
'''
### Shuffling the dataframe
dataFrame = ShufflingData(dataFrame)

### Saving number of events for each origin
for origin in inputOrigins:
    logFile.write('\nNumber of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]) + ' (raw), ' + str(sum(dataFrame[dataFrame['origin'] == origin]['weight'])) +' (with MC weights)')
    print(Fore.BLUE + 'Number of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]) + ' (raw), ' + str(sum(dataFrame[dataFrame['origin'] == origin]['weight'])) +' (with MC weights)')
#exit()
### Saving the combined dataframe
outputFileName = '/MixData_' + fileCommonName + '.pkl'
dataFrame.to_pickle(outputDir + outputFileName)
print(Fore.GREEN + 'Saved ' + outputDir + outputFileName)
logFile.write('\nSaved combined (signal and background) dataframe in ' + outputDir + outputFileName)

### Closing the log file
logFile.close()
print(Fore.GREEN + 'Saved ' + logFileName)

### Drawing histogram of variables
if drawPlots:
    histoOutputDir = outputDir + '/trainTestHistograms'
    checkCreateDir(histoOutputDir)
    DrawVariablesHisto(dataFrame, InputFeatures, histoOutputDir, fileCommonName, jetCollection, analysis, channel, signal, backgroundLegend, preselectionCuts, False)
    DrawCorrelationMatrix(dataFrame, InputFeatures, outputDir, fileCommonName, analysis, channel, signal, backgroundLegend)
