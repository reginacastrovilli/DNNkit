from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectRegime, CutMasses, defineBins
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
import math
import json
from matplotlib import gridspec

NN = 'PDNN'

### Reading the command line and extracting analysis and channel
tag, jetCollection, regime, preselectionCuts, signal, background = ReadArgParser()

### move inside the loop over the regimes if combining different analysis/channels
if 'Merg' in regime[0]:
    analysis = 'merged'
elif 'Res' in regime[0]:
    analysis = 'resolved'
if 'GGF' in regime[0]:
    channel = 'ggF'
elif 'VBF' in regime[0]:
    channel = 'VBF'
    signal = channel + signal

if len(regime) > 1:
    regimeString = '_'.join(regime)

### Reading from config file
inputFiles, rootBranchSubSample, InputFeatures, dfPath, variablesToSave, backgroundsList = ReadConfig(tag, analysis, jetCollection)

### Creating the list of the background origins selected
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

significanceRegimeScoresDict = {}
significanceRegimeMassDict = {}
combinedSignificanceSquareScoresDict = {}
combinedSignificanceSquareMassDict = {}

overwriteDataFrame = False
for regimeToTest in regime:
    if len(regime) > 1:
        print(Fore.RED + 'Selecting regime = ' + regimeToTest)
    significanceRegimeScoresDict[regimeToTest] = {}
    significanceRegimeMassDict[regimeToTest] = {}

    ### Loading signal and background dataframes if already existing
    inputOutputDir = dfPath + regimeToTest + '/'
    print (format('Output directory: ' + Fore.GREEN + inputOutputDir), checkCreateDir(inputOutputDir))

    ### Creating empty signal and background dataframe
    dataFrameSignal = []
    dataFrameBkg = []

    for target in targetOrigins:
        outputFileName = target + '_' + tag + '_' + jetCollection + '_' + regimeToTest + '_' + preselectionCuts + '.pkl'

        ### Loading dataframe if found and overwrite flag is false
        if not overwriteDataFrame:
            inputFileName = inputOutputDir + outputFileName
            if os.path.isfile(inputFileName):
                if target == signal:
                    print(Fore.GREEN + 'Found signal dataframe: loading ' + inputFileName)
                    dataFrameSignal = pd.read_pickle(inputFileName)
                else:
                    print(Fore.GREEN + 'Found background dataframe: loading ' + inputFileName)
                    dataFrameBkg.append(pd.read_pickle(inputFileName))

        if overwriteDataFrame or not os.path.isfile(inputFileName):#outputFileName):
            inputDir = dfPath + analysis + '/' + channel + '/' ### rivedere le directories
            ### Defining local dataframe (we might have found only one among many dataframes)
            partialDataFrameBkg = []
            for file in os.listdir(inputDir):############dfPath):
                if (all(x in file for x in [target, analysis, channel, '.pkl'])):
                    ### Loading input file
                    print(Fore.GREEN + 'Loading ' + inputDir + file)
                    inputDf = pd.read_pickle(inputDir + file)
                    ### Selecting events in the desired regime
                    inputDf = SelectRegime(inputDf, preselectionCuts, regimeToTest, channel)
                    if target == signal:
                        dataFrameSignal.append(inputDf)
                    else:
                        partialDataFrameBkg.append(inputDf)
                        
            ### Concatening and saving signal and background dataframes
            if target == signal:
                dataFrameSignal = pd.concat(dataFrameSignal, ignore_index = True)
                dataFrameSignal.to_pickle(inputOutputDir + outputFileName)
            elif target != signal:
                partialDataFrameBkg = pd.concat(partialDataFrameBkg, ignore_index = True)
                partialDataFrameBkg.to_pickle(inputOutputDir + outputFileName)
                ### Appending the local background dataframe to the final one
                dataFrameBkg.append(partialDataFrameBkg)
            print(Fore.GREEN + 'Saved ' + inputOutputDir + outputFileName)

    ### Concatening the global background dataframe
    dataFrameBkg = pd.concat(dataFrameBkg, ignore_index = True)

    ### Converting DSID to mass in the signal dataframe
    massesSignal = dataFrameSignal['DSID'].copy()
    DSIDsignal = np.array(list(set(list(dataFrameSignal['DSID']))))
    for DSID in DSIDsignal:
        massesSignal = np.where(massesSignal == DSID, DictDSID[DSID], massesSignal)
    dataFrameSignal = dataFrameSignal.assign(mass = massesSignal)

    ### Cutting signal events according to their mass and the type of analysis
    dataFrameSignal = CutMasses(dataFrameSignal, analysis)
    massesSignalList = list(dict.fromkeys(list(dataFrameSignal['mass'])))
    print(Fore.BLUE + 'Masses in the signal sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV')
    
    ### Saving MC weights
    signalMCweights = dataFrameSignal['weight']
    bkgMCweights = dataFrameBkg['weight']
    
    ### Saving X_boosted_m
    X_boosted_m_signal = dataFrameSignal['X_boosted_m']
    X_boosted_m_bkg = dataFrameBkg['X_boosted_m']

    ### Assigning a random mass to background events
    massesBkg = np.random.choice(massesSignalList, dataFrameBkg.shape[0])
    dataFrameBkg = dataFrameBkg.assign(mass = massesBkg)
    print('Number of background events: ' + str(dataFrameBkg.shape[0]))

    ### Selecting in the dataframe only the variables that will be used as input for the NN
    dataFrameSignal = dataFrameSignal[InputFeatures]
    dataFrameBkg = dataFrameBkg[InputFeatures]

    ### Scaling variables according to the variables.json file produced by the NN
    commonDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background + '_fullStat' + '/' ### fullStat provvisorio!!!
    modelDir = commonDir + NN + '/'
    variablesFile = modelDir + 'variables.json'
    jsonFile = open(variablesFile, 'r')
    print(Fore.GREEN + 'Loading ' + variablesFile)
    values = json.load(jsonFile)
    ### controllo su confronto variabili in df e json file!!!!!!!!!!!! TODO 
    for field in values['inputs']:
        feature = field['name']
        offset = field['offset']
        scale = field['scale']
        dataFrameSignal[feature] = (dataFrameSignal[feature] - offset) / scale
        dataFrameBkg[feature] = (dataFrameBkg[feature] - offset) / scale
    jsonFile.close()
    
    ### Assigning scaled to unscaled mass values
    scaledMassesSignalList = list(dict.fromkeys(list(dataFrameSignal['mass'])))
    massesDictionary = dict(zip(massesSignalList, scaledMassesSignalList))
    
    ### Loading model produced by the NN
    from keras.models import model_from_json
    architectureFile = modelDir + 'architecture.json'
    with open(architectureFile, 'r') as json_file:
        print(Fore.GREEN + 'Loading ' + architectureFile)
        model = model_from_json(''.join(json_file.readlines()))
    
    ### Loading weights into the model
    weightsFile = modelDir + 'weights.h5'
    model.load_weights(weightsFile)
    print(Fore.GREEN + 'Loading ' + weightsFile)

    Nbins = 100#40 ### change to variable bin size
    batchSize = 2048 ### uguale a quella del train?
    significanceDict = {}
    sortedMassList = np.sort(np.array(massesSignalList))
    for feature in ['Scores', 'Invariant mass']:
        print(feature)
        significanceList = []
        for mass in sortedMassList:
            scaledMass = massesDictionary[mass]
            dataFrameSignalMass = dataFrameSignal[dataFrameSignal['mass'] == scaledMass]
            signalMCweightsMass = signalMCweights[dataFrameSignal['mass'] == scaledMass]
            print('Number of signal events with mass ' + str(mass) + ' GeV: ' + str(dataFrameSignalMass.shape[0]))

            if feature == 'Scores':
                dataFrameBkg = dataFrameBkg.assign(mass = np.full(len(dataFrameBkg), scaledMass))
                hist_signal = model.predict(np.array(dataFrameSignalMass.values).astype(np.float32), batch_size = batchSize)
                hist_bkg = model.predict(np.array(dataFrameBkg.values).astype(np.float32), batch_size = batchSize)
                Bins = np.linspace(-0.05, 1.05, Nbins)
            elif feature == 'Invariant mass':
                hist_signal = X_boosted_m_signal[dataFrameSignal['mass'] == scaledMass]
                hist_bkg = X_boosted_m_bkg
                minEdge = min(min(hist_signal), min(hist_bkg))
                maxEdge = max(max(hist_signal), max(hist_bkg))
                #Bins = np.linspace(minEdge, maxEdge, Nbins)
                Bins = defineBins(regimeToTest)
            binContentsSignal, binEdgesSignal, _ = plt.hist(hist_signal, weights = signalMCweightsMass, bins = Bins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal'])#, density = True) ###### density???
            binContentsBkg, binEdgesBkg, _ = plt.hist(hist_bkg, weights = bkgMCweights, bins = Bins, histtype = 'step', lw = 2, color = 'red', label = [r'Background'])#, density = True)
            plt.xlabel(feature)
            plt.ylabel('Weighted counts')
            plt.yscale('log')
            legendText = 'Signal: ' + signal + ' (mass ' + str(mass) + ' GeV)\nBackground: ' + str(background) + '\nRegime: ' + regimeToTest
            plt.figtext(0.3, 0.7, legendText, wrap = True, horizontalalignment = 'left')
            plt.legend()
            if feature == 'Scores':
                pltName = inputOutputDir + feature + '_' + regimeToTest + '_' + str(mass) + '_' + tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '.png'
            elif feature == 'Invariant mass':
                pltName = inputOutputDir + 'InvariantMass_' + regimeToTest + '_' + str(mass) + '_' + tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '.png'
            plt.savefig(pltName)
            print(Fore.GREEN + 'Saved ' + pltName)
            '''
            if feature == 'Invariant mass':
                print(binContentsSignal)
                print(mass)
                plt.xlabel('X_boosted_m [GeV]')
                plt.ylabel('Weighted counts')
                plt.legend()
                #plt.savefig('InvariantMass_' + regimeToTest + '_' + str(mass) + '.png')
                plt.show()
            exit()
            '''
            ### Computing significance
            total = 0
            for eachBkgBinContent, eachSignalBinContent in zip(binContentsBkg, binContentsSignal):
                if eachBkgBinContent > 0 and eachSignalBinContent > 0:
                    total += 2 * ((eachSignalBinContent + eachBkgBinContent) * math.log(1 + eachSignalBinContent / eachBkgBinContent) - eachSignalBinContent)
            significanceMass = math.sqrt(total)
            significanceList.append(significanceMass)
            plt.clf()
            #if len(regime) > 1 and feature == 'Scores': 
            if len(regime) > 1:
                if feature == 'Scores':
                    significanceRegimeScoresDict[regimeToTest][mass] = significanceMass
                    if regimeToTest == regime[0]:
                        combinedSignificanceSquareScoresDict[mass] = 0
                    combinedSignificanceSquareScoresDict[mass] += significanceMass ** 2
                if feature == 'Invariant mass':
                    significanceRegimeMassDict[regimeToTest][mass] = significanceMass
                    if regimeToTest == regime[0]:
                        combinedSignificanceSquareMassDict[mass] = 0
                    combinedSignificanceSquareMassDict[mass] += significanceMass ** 2
        significanceDict[feature] = significanceList
            
    ### Plotting significance for scores and invariant mass and their ratio
    sortedMassListTeV = [float(mass / 1000) for mass in sortedMassList]
    fig = plt.figure(figsize = (8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1], hspace = 0.2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    legendText = 'jet collection: ' + jetCollection + '\nsignal: ' + signal + '\nbackground: ' + str(background) + '\nregime: ' + regimeToTest
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    colorsDict = {'Scores': 'red', 'Invariant mass': 'blue'}
    for key in significanceDict:
        ax1.plot(sortedMassListTeV, significanceDict[key], color = colorsDict[key], marker = 'o', label = key)
        ax2.plot(sortedMassListTeV, np.divide(np.array(significanceDict[key]), np.array(significanceDict['Invariant mass'])), color = colorsDict[key], marker = 'o')
    emptyPlot, = ax1.plot(sortedMassListTeV[0], significanceDict['Scores'][0], color = 'white')#, label = legendText)
    legend1 = ax1.legend(loc = 'upper left')
    legend2 = ax1.legend([emptyPlot], [legendText], frameon = True, handlelength = 0, handletextpad = 0, loc = 'lower right')
    ax1.add_artist(legend1)
    ax1.set(xlabel = 'Mass [TeV]')
    ax1.set(ylabel = 'Significance', yscale = 'log')
    ax2.set(xlabel = 'Mass [TeV]')
    ax2.set(ylabel = 'Ratio to invariant mass')
    plt.tight_layout()
    fileCommonName = tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '_' + regimeToTest
    pltName = inputOutputDir + 'Significance_' + fileCommonName + '.png'
    plt.savefig(pltName)
    print(Fore.GREEN + 'Saved ' + pltName)
    ax1.clear()
    ax2.clear()

if len(regime) > 1:
    plt.clf()
    legendText = 'jet collection: ' + jetCollection + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    legendText += '\nregimes:'
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    for regimeToTest in regime:
        legendText += '\n' + regimeToTest
    combinedSignificanceSqaureScoresValues = list(combinedSignificanceSquareScoresDict.values())
    combinedSignificanceScores = [math.sqrt(significance) for significance in combinedSignificanceSqaureScoresValues]
    combinedSignificanceSqaureMassValues = list(combinedSignificanceSquareMassDict.values())
    combinedSignificanceMass = [math.sqrt(significance) for significance in combinedSignificanceSqaureMassValues]
    fig = plt.figure(figsize = (8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1], hspace = 0.2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.plot(sortedMassListTeV, combinedSignificanceScores, marker = 'o', color = 'red', label = 'Scores')
    ax1.plot(sortedMassListTeV, combinedSignificanceMass, marker = 'o', color = 'blue', label = 'Invariant mass')
    emptyPlot, = ax1.plot(sortedMassListTeV[0], combinedSignificanceScores[0], color = 'white')#, label = legendText)
    legend1 = ax1.legend(loc = 'upper left')
    legend2 = ax1.legend([emptyPlot], [legendText], frameon = True, handlelength = 0, handletextpad = 0, loc = 'lower right')
    ax1.add_artist(legend1)
    ax2.plot(sortedMassListTeV, np.divide(np.array(combinedSignificanceScores), np.array(combinedSignificanceMass)), color = 'red', marker = 'o')
    ax2.plot(sortedMassListTeV, np.divide(np.array(combinedSignificanceMass), np.array(combinedSignificanceMass)), color = 'blue', marker = 'o')
    ax1.set(xlabel = 'Mass [TeV]')
    ax1.set(ylabel = 'Significance', yscale = 'log')
    ax2.set(xlabel = 'Mass [TeV]')
    ax2.set(ylabel = 'Ratio to invariant mass')
    pltName = commonDir + 'CombinedSignificance_' + tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + regimeString + '.png'
    plt.savefig(pltName)
    print(Fore.GREEN + 'Saved ' + pltName)

