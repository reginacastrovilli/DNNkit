from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData, SelectRegime, CutMasses, defineBins, weighted_percentile, defineFixBins, defineVariableBins, sortColumns, defineBinsNew, scaleVariables, loadModelAndWeights
from sklearn.metrics import roc_curve, auc, roc_auc_score
import atlasplots as aplt
import ROOT as root
import pandas as pd
import numpy as np
import re
import ast
import os.path
from colorama import init, Fore, Back
init(autoreset = True)
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib import gridspec
import seaborn
import shutil

saveResults = True

#aplt.set_atlas_style()
NN = 'PDNN'

### Reading the command line and extracting analysis and channel
tag, jetCollection, regime, preselectionCuts, signalLabel, background = ReadArgParser()

### move inside the loop over the regimes if combining different analysis/channels
if 'Merg' in regime[0]:
    analysis = 'merged'
    lowerHistoMassEdge = 500
    upperHistoMassEdge = 3500
elif 'Res' in regime[0]:
    analysis = 'resolved'
    #lowerHistoMassEdge = 400 ## ??? TODO (signal dependent?)
    #upperHistoMassEdge = 4000 ## ??? TODO (signal dependent?)

if 'GGF' in regime[0]:
    channel = 'ggF'
    signal = signalLabel
    if 'WZ' in signalLabel:
        signalLabel = 'HVT'

elif 'VBF' in regime[0]:
    channel = 'VBF'
    signal = channel + signalLabel
    if 'WZ' in signalLabel:
        signalLabel = 'HVT'
if 'allMerged' in regime[0]:
    channel = 'ggF'
    signalLabel = signal
    analysis = 'merged'

if len(regime) > 1:
    regimeString = '_'.join(regime)

### Reading from config file
inputFiles, rootBranchSubSample, inputFeatures, dfPath, variablesToSave, backgroundsList = ReadConfig(tag, analysis, jetCollection, signal)

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

overwriteDataFrame = False
featuresToPlot = ['InvariantMass', 'Scores']
#significanceListDict = {}
significanceDict = {}
colorsDict = {'Scores': 'red', 'InvariantMass': 'blue'}

for regimeToTest in regime:
    if len(regime) > 1:
        print(Fore.RED + 'Selecting regime = ' + regimeToTest)

    ### Loading signal and background dataframes if already existing
    inputDir = dfPath + preselectionCuts + '/' + regimeToTest + '/'
    #outputDir = inputDir + 'tmp/withX_boosted_m/'# + 'loosePDNN/' <-------------------- OK
    #outputDir = inputDir + 'withoutDNNscore/ggFpDNN/'# + 'loosePDNN/' <-------------------- OK
    #outputDir = inputDir + 'ggFVBF/'
    #outputDir = inputDir + 'VBFggF/'
    outputDir = inputDir + signal + '/' + 'withoutDNNscore/'# + 'loosePDNN/' <-------------------- OK
    #outputDir = dfPath + preselectionCuts + '/deepPDNN/' + regimeToTest + '/'
    print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
    if saveResults:
        fileCommonName = tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '_' + regimeToTest
        logFileName = outputDir + 'logFile_computeSignificance_' + fileCommonName + '.txt'
        logFile = open(logFileName, 'w')

    ### Creating empty signal and background dataframe
    dataFrameSignal = []
    dataFrameBkg = []

    for target in targetOrigins:
        outputFileName = target + '_' + tag + '_' + jetCollection + '_' + regimeToTest + '_' + preselectionCuts + '.pkl'

        ### Loading dataframe if found and overwrite flag is false
        if not overwriteDataFrame:
            inputFileName = inputDir + outputFileName
            if os.path.isfile(inputFileName):
                if target == signal:
                    print(Fore.GREEN + 'Found signal dataframe: loading ' + inputFileName)
                    dataFrameSignal = pd.read_pickle(inputFileName)
                    if saveResults:
                        logFile.write('Found signal dataframe: loading ' + inputFileName + '\n')
                else:
                    print(Fore.GREEN + 'Found background dataframe: loading ' + inputFileName)
                    dataFrameBkg.append(pd.read_pickle(inputFileName))
                    if saveResults:
                        logFile.write('Found background dataframe: loading ' + inputFileName + '\n')

        if overwriteDataFrame or not os.path.isfile(inputFileName):
            inputDir = dfPath + analysis + '/' + channel + '/none/'
            ### Defining local dataframe (we might have found only one among many dataframes)
            partialDataFrameBkg = []
            print(target)
            for file in os.listdir(inputDir):
                if (all(x in file for x in [target, analysis, channel, 'none', '.pkl'])):
                    if 'VBF' in file and 'VBF' not in channel:#signal:
                        continue
                    ### Loading input file
                    print(Fore.GREEN + 'Loading ' + inputDir + file)
                    inputDf = pd.read_pickle(inputDir + file)
                    if saveResults:
                        logFile.write('Loading ' + inputFileName + '\n')

                    ### Selecting events in the desired regime
                    inputDf = SelectRegime(inputDf, preselectionCuts, regimeToTest, channel)
                    if target == signal:
                        dataFrameSignal.append(inputDf)
                    else:
                        partialDataFrameBkg.append(inputDf)
                    
            ### Concatening and saving signal and background dataframes
            if target == signal:
                dataFrameSignal = pd.concat(dataFrameSignal, ignore_index = True)
                dataFrameSignal.to_pickle(inputDir + outputFileName)
            elif target != signal:
                partialDataFrameBkg = pd.concat(partialDataFrameBkg, ignore_index = True)
                partialDataFrameBkg.to_pickle(inputDir + outputFileName)
                ### Appending the local background dataframe to the final one
                dataFrameBkg.append(partialDataFrameBkg)
            print(Fore.GREEN + 'Saved ' + inputDir + outputFileName)
            if saveResults:
                logFile.write('Saved ' + inputFileName + '\n')

    ### Concatening the global background dataframe, saving origin column and list, creating a new dummy 'mass' column filled with 0
    dataFrameBkg = pd.concat(dataFrameBkg, ignore_index = True)

    ### Removing events with high absoulte MC weights
    dataFrameBkg = dataFrameBkg[dataFrameBkg['weight'] > -15]
    backgroundsList = list(set(list(dataFrameBkg['origin'])))
    originsBkg = dataFrameBkg['origin']
    print('Number of background events: ' + str(dataFrameBkg.shape[0]) + ' raw, ' + str(dataFrameBkg['weight'].sum()) + ' with MC weights')
    if saveResults:
        logFile.write('Number of background events: ' + str(dataFrameBkg.shape[0]) + ' raw, ' + str(dataFrameBkg['weight'].sum()) + ' with MC weights\n')
    dataFrameBkg = dataFrameBkg.assign(mass = np.zeros(len(dataFrameBkg)))

    ### Converting DSID to mass in the signal dataframe
    massesSignal = dataFrameSignal['DSID'].copy()
    DSIDsignal = np.array(list(set(list(dataFrameSignal['DSID']))))
    for DSID in DSIDsignal:
        massesSignal = np.where(massesSignal == DSID, DictDSID[DSID], massesSignal)
    dataFrameSignal = dataFrameSignal.assign(mass = massesSignal)
    '''
    ### tmp
    invariant_mass_signal = dataFrameSignal['X_boosted_m'].copy()
    invariant_mass_bkg = dataFrameBkg['X_boosted_m'].copy()
    dataFrameSignal = dataFrameSignal.assign(X_boosted_m_unscaled = np.array(invariant_mass_signal))
    dataFrameBkg = dataFrameBkg.assign(X_boosted_m_unscaled = np.array(invariant_mass_bkg))
    '''
    ### Saving only variables that will be used as input to the neural network, X_boosted_m, weight
    columnsToSave = inputFeatures + ['origin', 'X_boosted_m', 'weight']
    #columnsToSave = inputFeatures + ['origin', 'weight', 'X_boosted_m_unscaled'] ###tmp
    dataFrameSignal = dataFrameSignal[columnsToSave]
    dataFrameBkg = dataFrameBkg[columnsToSave]

    '''
    dataFrameSignal1200 = dataFrameSignal[dataFrameSignal['mass'] == 1200]
    print(dataFrameSignal1200)
    print(dataFrameSignal1200['X_boosted_m'])
    plt.hist(dataFrameSignal1200['X_boosted_m'], weights = dataFrameSignal1200['weight'], bins = np.linspace(900, 1600, 101), label = ['Signal'])
    plt.hist(dataFrameBkg['X_boosted_m'], weights = dataFrameBkg['weight'], bins = np.linspace(900, 1600, 101), label = ['Bkg'])
    plt.legend()
    plt.show()
    exit()
    '''
    ### Cutting signal events according to their mass and the type of analysis
    dataFrameSignal = CutMasses(dataFrameSignal, analysis)
    massesSignalList = list(dict.fromkeys(list(dataFrameSignal['mass'])))
    print(Fore.BLUE + 'Number of signal (' + signal + ') events: ' + str(dataFrameSignal.shape[0]) + ' raw, ' + str(dataFrameSignal['weight'].sum()) + ' with MC weights')
    print(Fore.BLUE + 'Masses in the signal (' + signal + ') sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV')
    if saveResults:
        logFile.write('Number of signal (' + signal + ') events: ' + str(dataFrameSignal.shape[0]) + ' raw, ' + str(dataFrameSignal['weight'].sum()) + ' with MC weights\n')
        logFile.write('Masses in the signal (' + signal + ') sample: ' + str(np.sort(np.array(massesSignalList))) + ' GeV\n')

    ### Scaling variables according to the variables.json file produced by the NN
    #modelDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/' + NN + '/' ##### <--------- OK
    #modelDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/tmp/' + NN + '/withoutDNNscore/' ##### <--------- OK
    modelDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/' + NN + '/withoutDNNscore/' ##### <--------- OK
    #modelDir = '/nfs/kloe/einstein4/HDBS/NNoutput/r33-24/UFO_PFLOW/merged/ggF/none/ggFVBF/Radion/all/PDNN/'
    #modelDir = '/nfs/kloe/einstein4/HDBS/NNoutput/r33-24/UFO_PFLOW/merged/VBF/none/VBFRSG/all/PDNN/withoutDNNscore/'
    dataFrameSignal, dataFrameBkg = scaleVariables(modelDir, dataFrameSignal, dataFrameBkg, inputFeatures, outputDir)

    ### Assigning scaled to unscaled mass values
    scaledMassesSignalList = list(dict.fromkeys(list(dataFrameSignal['mass'])))
    massesDictionary = dict(zip(massesSignalList, scaledMassesSignalList))

    #modelDir = '/nfs/kloe/einstein4/HDBS/NNoutput/r33-24/UFO_PFLOW/merged/ggF/none/HVTWZ/all/PDNN/withoutDNNscore/'
    #print(Fore.RED + 'New modelDir' + modelDir)

    ### Loading model produced by the NN
    model, batchSize = loadModelAndWeights(modelDir, outputDir)

    ### Copying logFile from the modelDir directory to the output one
    for file in os.listdir(modelDir):
        if 'logFile' in file:
            shutil.copyfile(modelDir + file, outputDir + file)
            print(Fore.GREEN + 'Copied input logFile to ' + outputDir + file)

    sortedMassList = np.sort(np.array(massesSignalList))
    lowerHistoMassEdge = max(lowerHistoMassEdge, sortedMassList[1])
    upperHistoMassEdge = min(upperHistoMassEdge, sortedMassList[len(sortedMassList) - 2])
    significantBinsDict = {}
    massesToPlotTeV = np.array([])

    nBinsMassDict = {}
    bkgEventsInResolutionDict = {}
    bkgEventsInResolution = 0
    bkgEventsInMassResolutionDict = {}
    
    ### Saving the invariant mass distribution for the background
    invariantMassBkg = dataFrameBkg['X_boosted_m']
    #invariantMassBkg = dataFrameBkg['X_boosted_m_unscaled'] ###tmp
    sortedBkgEvents, sortedBkgEventsWeights = sortColumns(invariantMassBkg, dataFrameBkg['weight'])                    
    
    significanceDict[regimeToTest] = {}
    for feature in featuresToPlot:
        significanceDict[regimeToTest][feature] = {}

    for mass in sortedMassList:
        '''
        if mass != 1500 and mass != 1000:
            continue
        '''
        if mass < lowerHistoMassEdge or mass > upperHistoMassEdge:
            continue
        print(Fore.CYAN + '-------------- ' + str(mass) + ' -------------') 
        massesToPlotTeV = np.append(massesToPlotTeV, float(mass / 1000))

        ### Assigning the unscaled mass to the scaled one
        scaledMass = massesDictionary[mass]

        ### Selecting signal events with the desired mass and saving their MC weights 
        dataFrameSignalMass = dataFrameSignal[dataFrameSignal['mass'] == scaledMass]
        signalMCweightsMass = dataFrameSignalMass['weight'] # * 0.001
        print(Fore.BLUE + 'Number of signal (' + signal + ') events with mass ' + str(mass) + ' GeV: ' + str(dataFrameSignalMass.shape[0]) + ' raw, ' + str(dataFrameSignalMass['weight'].sum()) + ' with MC weights')

        for feature in featuresToPlot:
            print(Fore.CYAN + '########### ' + feature + ' ###########')
            if feature == 'InvariantMass':
                hist_signal = dataFrameSignalMass['X_boosted_m']
                #hist_signal = dataFrameSignalMass['X_boosted_m_unscaled'] ###tmp
                resolutionRangeLeft, resolutionRangeRight = weighted_percentile(hist_signal, signalMCweightsMass, feature)
                Bins, bkgEventsInResolutionDict[mass] = defineVariableBins(sortedBkgEvents, sortedBkgEventsWeights, lowerHistoMassEdge, upperHistoMassEdge, resolutionRangeLeft, resolutionRangeRight, feature, 1, 1)
                '''
                hist_signal = 1 - (abs(hist_signal - resolutionRangeLeft) / diff)
                BinsInvariantMass = defineBinsNew(hist_signal, signalMCweightsMass)
                '''
                prediction = 'X_boosted_m'
                #prediction = 'X_boosted_m_unscaled' ###tmp
                featureLabel = 'Invariant mass [GeV]'
                hist_bkg = invariantMassBkg
                #Bins = BinsInvariantMass
                nBinsMassDict[mass] = len(Bins) - 1
                '''
                hist_bkg = 1 - (abs(dataFrameBkg['X_boosted_m'] - resolutionRangeLeft) / diff)
                dataFrameBkgScaled = dataFrameBkg.copy()
                dataFrameBkgScaled = dataFrameBkgScaled.assign(X_boosted_m = hist_bkg)
                '''

            elif feature == 'Scores':
                ### Prediction on signal
                hist_signal = model.predict(np.array(dataFrameSignalMass[inputFeatures].values).astype(np.float32), batch_size = batchSize)
                hist_signal2 = np.array([])
                for i in range(len(hist_signal)):
                    hist_signal2 = np.append(hist_signal2, hist_signal[i]) ### TODO trovare un modo per risolvere!
                ### Computing the resolution of the signal distribution
                #hist_signal2 = (np.exp(hist_signal2) - 1) / np.exp(1)
                resolutionRangeLeft, resolutionRangeRight = weighted_percentile(hist_signal2, signalMCweightsMass, feature)
                ### Assigning the mass hypothesis background
                dataFrameBkg = dataFrameBkg.assign(mass = np.full(len(dataFrameBkg), scaledMass))
                ### Prediction on background
                hist_bkg = model.predict(np.array(dataFrameBkg[inputFeatures].values).astype(np.float32), batch_size = batchSize)
                #hist_bkg = (np.exp(hist_bkg) - 1) / np.exp(1)
                ### Computing the bins
                Bins = defineVariableBins(hist_bkg, dataFrameBkg['weight'], 0, 1, resolutionRangeLeft, resolutionRangeRight, feature, nBinsMassDict[mass], bkgEventsInResolutionDict[mass])
                '''
                hist_signal2 = (hist_signal2 - min(hist_signal2)) / (1 - min(hist_signal2))
                hist_bkg = (hist_bkg - min(hist_signal2)) / (1 - min(hist_signal2))
                BinsScores = defineBinsNew(hist_signal2, signalMCweightsMass)
                Bins = BinsScores
                print(Bins)
                '''
                
                ### Creating a new column in the background dataframes with the scores
                dataFrameBkg = dataFrameBkg.assign(scores = hist_bkg)
                prediction = 'scores'
                featureLabel = feature

            originsNumber = len(list(set(list(dataFrameBkg['origin']))))
            bkgPlot = seaborn.histplot(data = dataFrameBkg[prediction], x = dataFrameBkg[prediction], weights = dataFrameBkg['weight'], bins = Bins, hue = dataFrameBkg['origin'], multiple = 'stack')
            binHeights = []
            for rectangle in bkgPlot.patches:
                binHeights.append(rectangle.get_height())
            binHeights = np.array(binHeights).reshape(originsNumber, int(len(binHeights)/originsNumber))
            binContentsBkg = sum(binHeights)
            print('binContentsBkg:', binContentsBkg)

            binContentsSignal, binEdgesSignal, _ = plt.hist(hist_signal, weights = signalMCweightsMass, bins = Bins, histtype = 'step', lw = 2, color = 'red', label = [r'Signal']) #<---------------------
            print('binContentsSignal:', binContentsSignal)

            if saveResults:
                plt.legend() ###
                plt.xlabel(featureLabel)
                plt.ylabel('Weighted counts')
                #plt.xscale('log')
                plt.yscale('log')
                plt.title(signalLabel + ' ' + str(mass) + ' GeV, ' + regimeToTest)
                pltName = outputDir + feature + '_' + regimeToTest + '_' + str(mass) + '_' + tag + '_' + jetCollection + '_' + preselectionCuts + '_' + signal + '_' + background + '.png'
                plt.savefig(pltName)
                print(Fore.GREEN + 'Saved ' + pltName)
            plt.clf()

            ### Computing significance
            total = 0
            delta = 0.3
            for b, s in zip(binContentsBkg, binContentsSignal):
                total += s * s / (s + 1.3 * b)
                '''
                if b > 0 and s > 0:
                    #s *= 1e-3
                    sigPlusBkg = s + b
                    sigma = (delta * b) ** 2
                    first = sigPlusBkg * math.log(sigPlusBkg * (b + sigma) / (b ** 2 + sigPlusBkg * sigma))
                    second = (b ** 2) / sigma * math.log(1 + sigma * s / (b * (b + sigma)))
                    total += 2 * (first - second) ### it requires b != 0
                    #total += 2 * ((s + b) * math.log(1 + s / b) - b)
                    #total += 2 * (sigPlusBkg * math.log((sigPlusBkg * bkgPlusError) / (eachBkgBinContent ** 2 + sigPlusBkg * bkgPlusError)) - (eachBkgBinContent ** 2) / (bkgPlusError ** 2) * math.log(1 + (eachSignalBinContent * bkgPlusError ** 2) / eachBkgBinContent * (eachBkgBinContent + bkgPlusError ** 2))
                    #total += 2 * ((eachSignalBinContent + eachBkgBinContent) * math.log((eachSignalBinContent + eachBkgBinContent) * (eachBkgBinContent + (0 * eachBkgBinContent) ** 2) / (eachBkgBinContent ** 2 + (eachSignalBinContent + eachBkgBinContent) * (0 * eachBkgBinContent) ** 2)) - (eachBkgBinContent ** 2)/((0 * eachBkgBinContent) ** 2) * math.log(1 + (0 * eachBkgBinContent) ** 2 * eachSignalBinContent / eachBkgBinContent * (eachBkgBinContent + (0 * eachBkgBinContent) ** 2)))
                    #total += (eachSignalBinContent / math.sqrt(eachSignalBinContent + 1.3 * eachBkgBinContent)) ** 2

                else:
                    print(Back.RED + 'IMPOSSIBLE TO COMPUTE SIGNIFICANCE!')
                '''
            significanceMass = math.sqrt(total)
            print(Fore.RED + 'Significance for ' + feature + ': ' + str(significanceMass))
            significanceDict[regimeToTest][feature][mass] = significanceMass

    if not saveResults:
        exit()

    logFile.close()

    ### Plotting significance for scores and invariant mass and their ratio
    fig = plt.figure(figsize = (8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1], hspace = 0.2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    legendText = 'jet collection: ' + jetCollection + '\nsignal: ' + signalLabel + '\nbackground: ' + str(background) + '\nregime: ' + regimeToTest + '\nProduction: ' + tag + '\nwithout DNN scores'#\nwith pDNN trained on ggF'#'\nwith X_boosted_m'
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts

    for feature in featuresToPlot:
        xValues, yValues = zip(*significanceDict[regimeToTest][feature].items())
        xValues = np.array(xValues) / 1000
        ax1.plot(xValues, yValues, color = colorsDict[feature], marker = 'o', label = feature)
        ratioDict = {key: significanceDict[regimeToTest][feature][key] / significanceDict[regimeToTest]['InvariantMass'].get(key, 0) for key in significanceDict[regimeToTest]['InvariantMass'].keys()}
        ratioValues = ratioDict.values()
        ax2.plot(xValues, ratioValues, color = colorsDict[feature], marker = 'o')

    fakeX = list(significanceDict[regimeToTest]['Scores'].keys())[0]
    fakeY = list(significanceDict[regimeToTest]['Scores'].values())[0]
    emptyPlot, = ax1.plot(fakeX / 1000, fakeY, color = 'white')#, label = legendText)
    legend1 = ax1.legend(loc = 'upper left')
    legend2 = ax1.legend([emptyPlot], [legendText], frameon = True, handlelength = 0, handletextpad = 0, loc = 'lower right')
    ax1.add_artist(legend1)
    ax1.set(xlabel = 'Mass [TeV]')
    ax1.set(ylabel = 'Significance')#, yscale = 'log')
    ax2.set(xlabel = 'Mass [TeV]')
    ax2.set(ylabel = 'Ratio to invariant mass')
    ax2.set(ylim = (0.5, 2))
    plt.tight_layout()
    pltName = outputDir + 'Significance_' + fileCommonName + '.png'
    plt.savefig(pltName)
    print(Fore.GREEN + 'Saved ' + pltName)
    ax1.clear()
    ax2.clear()

if len(regime) > 1:
    plt.clf()
    fig = plt.figure(figsize = (8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1], hspace = 0.2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    combinedSignificanceDict = {}
    for feature in featuresToPlot:
        combinedSignificanceDict[feature] = {}
        for mass in significanceDict[regime[0]]['Scores'].keys():
            combinedSignificanceDict[feature][mass] = 0
            for regimeToTest in regime:
                combinedSignificanceDict[feature][mass] += (significanceDict[regimeToTest][feature][mass] ** 2)
                if regimeToTest == regime[len(regime) - 1]:
                    combinedSignificanceDict[feature][mass] = math.sqrt(combinedSignificanceDict[feature][mass])

        xValues, yValues = zip(*combinedSignificanceDict[feature].items())
        xValues = np.array(xValues) / 1000
        ax1.plot(xValues, yValues, color = colorsDict[feature], marker = 'o', label = feature)
        ratioDict = {key: combinedSignificanceDict[feature][key] / combinedSignificanceDict['InvariantMass'].get(key, 0) for key in combinedSignificanceDict['InvariantMass'].keys()}
        ratioValues = ratioDict.values()
        ax2.plot(xValues, ratioValues, color = colorsDict[feature], marker = 'o')

    legendText = 'jet collection: ' + jetCollection + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    legendText += '\nregimes:'
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    for regimeToTest in regime:
        legendText += '\n' + regimeToTest
    fakeX = list(combinedSignificanceDict['Scores'].keys())[0]
    fakeY = list(combinedSignificanceDict['Scores'].values())[0]
    emptyPlot, = ax1.plot(fakeX / 1000, fakeY, color = 'white')#, label = legendText)
    legend1 = ax1.legend(loc = 'upper left')
    legend2 = ax1.legend([emptyPlot], [legendText], frameon = True, handlelength = 0, handletextpad = 0, loc = 'lower right')
    ax1.add_artist(legend1)
    ax1.set(xlabel = 'Mass [TeV]')
    ax1.set(ylabel = 'Combined significance')#, yscale = 'log')
    ax2.set(xlabel = 'Mass [TeV]')
    ax2.set(ylabel = 'Ratio to invariant mass')
    ax2.set(ylim = (0.5, 2))
    pltName = 'CombinedSignificance_' + tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + regimeString + '.png'
    plt.savefig(outputDir + pltName)
    print(Fore.GREEN + 'Saved ' + outputDir + pltName)
