from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectEvents, CutMasses, DrawVariablesHisto, DrawCorrelationMatrix
import pandas as pd
import numpy as np
import random
import os.path
from colorama import init, Fore
init(autoreset = True)

drawPlots = False
overwriteDataFrame = False

### Reading the command line
tag, jetCollection, analysis, channel, preselectionCuts, signal, background = ReadArgParser()

### Reading from config file
InputFeatures, dfPath, variablesToSave, backgroundsList = ReadConfig(tag, analysis, jetCollection)

### Creating output directories and logFile
tmpOutputDir = dfPath + analysis + '/' + channel
print(format('First output directory: ' + Fore.GREEN + tmpOutputDir), checkCreateDir(tmpOutputDir))
tmpFileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts

outputDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background
print(format('Second output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
fileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
logFileName = outputDir + '/EventsNumberAfterSelection_' + fileCommonName + '.txt'
logFile = open(logFileName, 'w')
logFile.write('CxAOD tag: ' + tag + '\nJetCollection: ' + jetCollection + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + '\nSignal:' + signal + '\nBackground: ' + background + '\nInput files path: ' + dfPath)# + '\nInput files and number of events after selection:\n')

### Loading DSID-mass map and storing it into a dictionary
DictDSID = {}
DSIDfile = open('DSIDtoMass.txt')
lines = DSIDfile.readlines()
for line in lines:
    DictDSID[int(line.split(':')[0])] = int(line.split(':')[1])

### Creating the list of the origins selected (signal + background)
if background == 'all':
    originsBkgTest = backgroundsList.copy()
else:
    originsBkgTest = list(background.split('_'))

targetOrigins = originsBkgTest.copy()
targetOrigins.insert(0, signal)

### Creating empty signal and background dataframe 
dataFrameSignal = []
dataFrameBkg = []

for target in targetOrigins:
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
                print(Fore.GREEN + 'Loading ' + dfPath + file)
                inputDf = pd.read_pickle(dfPath + file)
                '''
                print(inputDf.shape)
                inputDf1 = inputDf.query('Pass_MergHP_GGF_ZZ_Tag_SR == False and Pass_MergHP_GGF_ZZ_Untag_SR == False and Pass_MergHP_GGF_WZ_SR == False and Pass_MergLP_GGF_ZZ_Tag_SR == False and Pass_MergLP_GGF_ZZ_Untag_SR == False and Pass_MergLP_GGF_WZ_SR == False and Pass_MergHP_GGF_ZZ_Tag_ZCR == False and Pass_MergHP_GGF_ZZ_Untag_ZCR == False and Pass_MergHP_GGF_WZ_ZCR == False and Pass_MergLP_GGF_ZZ_Tag_ZCR == False and Pass_MergLP_GGF_ZZ_Untag_ZCR == False and Pass_MergLP_GGF_WZ_ZCR == False and Pass_MergHP_GGF_ZZ_Untag_TCR == False and Pass_MergHP_GGF_ZZ_Tag_TCR == False and Pass_MergHP_GGF_WZ_TCR == False and Pass_MergLP_GGF_ZZ_Untag_TCR == False and Pass_MergLP_GGF_ZZ_Tag_TCR == False and Pass_MergLP_GGF_WZ_TCR == False and Pass_Res_GGF_WZ_SR == False and Pass_Res_GGF_ZZ_Tag_SR == False and Pass_Res_GGF_ZZ_Untag_SR == False and Pass_Res_GGF_WZ_ZCR == False and Pass_Res_GGF_ZZ_Tag_ZCR == False and Pass_Res_GGF_ZZ_Untag_ZCR == False and Pass_Res_GGF_WZ_TCR == False and Pass_Res_GGF_ZZ_Tag_TCR == False and Pass_Res_GGF_ZZ_Untag_TCR == False and Pass_MergHP_VBF_WZ_SR == False and Pass_MergHP_VBF_ZZ_SR == False and Pass_MergLP_VBF_WZ_SR == False and Pass_MergLP_VBF_ZZ_SR == False and Pass_MergHP_VBF_WZ_ZCR == False and Pass_MergHP_VBF_ZZ_ZCR == False and Pass_MergLP_VBF_WZ_ZCR == False and Pass_MergLP_VBF_ZZ_ZCR == False and Pass_MergHP_VBF_WZ_TCR == False and Pass_MergHP_VBF_ZZ_TCR == False and Pass_MergLP_VBF_WZ_TCR == False and Pass_MergLP_VBF_ZZ_TCR == False and Pass_Res_VBF_WZ_SR == False and Pass_Res_VBF_ZZ_SR == False and Pass_Res_VBF_WZ_ZCR == False and Pass_Res_VBF_ZZ_ZCR == False and Pass_Res_VBF_WZ_TCR == False and Pass_Res_VBF_ZZ_TCR == False')
                print(inputDf1.shape)       
                exit()
                '''
                ### Selecting events according to merged/resolved regime and ggF/VBF channel
                inputDf = SelectEvents(inputDf, channel, analysis, preselectionCuts)
                #weighted_sum = inputDf['weight'].sum()
                #print('weighted sum: ' + str(weighted_sum))
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
#logFile.write('\nMasses in the signal sample: ' + str(np.sort(np.array(massesSignalList))))

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

### Saving number of events for each origin
for origin in targetOrigins:
    logFile.write('\nNumber of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]))
    print(Fore.BLUE + 'Number of ' + origin + ' events: ' + str(dataFrame[dataFrame['origin'] == origin].shape[0]))
    
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
    DrawVariablesHisto(dataFrame, InputFeatures, histoOutputDir, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts)
    DrawCorrelationMatrix(dataFrame, InputFeatures, outputDir, fileCommonName, analysis, channel, signal, background)

