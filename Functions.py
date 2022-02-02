# Assigning script names to variables
fileName1 = 'saveToPkl.py'
fileName2 = 'buildDataset.py'
fileName3 = 'splitDataset.py'
fileName4 = 'buildDNN.py'
#fileName5 = 'buildPDNNtuningHyp.py'
fileName5 = 'buildPDNN.py'
#fileName6 = 'tuningHyperparameters.py'
fileNamePlots = 'drawPlots.py'

### Reading the command line
from argparse import ArgumentParser
import sys
from colorama import init, Fore
init(autoreset = True)

def ReadArgParser():
    parser = ArgumentParser()
    parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
    parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
    parser.add_argument('-s', '--Signal', help = 'Signal: \'VBFHVTWZ\', \'Radion\', \'RSG\' or \'VBFRadion\'', type = str)
    parser.add_argument('-j', '--JetCollection', help = 'Jet collection: \'TCC\', \'UFO_PFLOW\'', type = str, default = 'UFO_PFLOW')
    parser.add_argument('-b', '--Background', help = 'Background: \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\' (in quotation mark separated by a space)', type = str, default = 'all')
    parser.add_argument('-t', '--TrainingFraction', help = 'Relative size of the training sample, between 0 and 1', default = 0.8)
    parser.add_argument('-p', '--PreselectionCuts', help = 'Preselection cut', type = str)
    parser.add_argument('-n', '--Nodes', help = 'Number of nodes of the (p)DNN, should always be >= nColumns and strictly positive', default = 128) #32
    parser.add_argument('-l', '--Layers', help = 'Number of hidden layers of the (p)DNN', default = 4) #2
    parser.add_argument('-e', '--Epochs', help = 'Number of epochs for the training', default = 200) #150
    parser.add_argument('-v', '--Validation', help = 'Fraction of the training data that will actually be used for validation', default = 0.2)
    parser.add_argument('-d', '--Dropout', help = 'Fraction of the neurons to drop during the training', default = 0.2)
    parser.add_argument('-m', '--Mass', help = 'Masses for the (P)DNN train/test (GeV, in quotation mark separated by a space)', default = 'all')
    parser.add_argument('--doTrain', help = 'If 1 the training will be performed, if 0 it won\'t', default = 1)
    parser.add_argument('--doTest', help = 'If 1 the test will be performed, if 0 it won\'t', default = 1)
    parser.add_argument('--loop', help = 'How many times the code will be executed', default = 1)
    parser.add_argument('--tag', help = 'CxAOD tag', default = 'r33-22')
    
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
    signal = args.Signal
    if args.Signal is None and sys.argv[0] != fileName1:
        parser.error(Fore.RED + 'Requested type of signal (\'Radion\', )')
    jetCollection = args.JetCollection
    if args.JetCollection is None:
        parser.error(Fore.RED + 'Requested jet collection (\'TCC\' or \'UFO_PFLOW\')')
    elif args.JetCollection != 'TCC' and args.JetCollection != 'UFO_PFLOW':
        parser.error(Fore.RED + 'Jet collection can be \'TCC\', \'UFO_PFLOW\'')
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
    doTrain = int(args.doTrain)
    if args.doTrain and (doTrain != 0 and doTrain != 1):
        parser.error(Fore.RED + 'doTrain can only be 1 (to perform the training) or 0')
    doTest = int(args.doTest)
    if args.doTest and (doTest != 0 and doTest != 1):
        parser.error(Fore.RED + 'doTest can only be 1 (to perform the test) or 0')
    loop = int(args.loop)
    tag = args.tag

    if sys.argv[0] == fileName1:
        print(Fore.BLUE + '           tag = ' + tag)
        print(Fore.BLUE + 'jet collection = ' + jetCollection)
        return tag, jetCollection

    if sys.argv[0] == fileName2:
        print(Fore.BLUE + '          background(s) = ' + str(backgroundString))
        print(Fore.BLUE + '                 signal = ' + str(signal))
        return tag, jetCollection, analysis, channel, preselectionCuts, signal, backgroundString

    if sys.argv[0] == fileName3 or sys.argv[0] == fileNamePlots:
        print(Fore.BLUE + '       background = ' + str(backgroundString))
        print(Fore.BLUE + 'training fraction = ' + str(trainingFraction))
        return tag, jetCollection, analysis, channel, preselectionCuts, backgroundString, signal, trainingFraction

    if(sys.argv[0] == fileName4 or sys.argv[0] == fileName5 or sys.argv[0] == fileName6):
        print(Fore.BLUE + '          background(s) = ' + str(backgroundString))
        print(Fore.BLUE + '          test mass(es) = ' + str(mass))
        print(Fore.BLUE + '      training fraction = ' + str(trainingFraction))
        print(Fore.BLUE + '        number of nodes = ' + str(numberOfNodes))
        print(Fore.BLUE + 'number of hidden layers = ' + str(numberOfLayers))
        print(Fore.BLUE + '       number of epochs = ' + str(numberOfEpochs))
        print(Fore.BLUE + '    validation fraction = ' + str(validationFraction))
        print(Fore.BLUE + '                dropout = ' + str(dropout))
        print(Fore.BLUE + '                doTrain = ' + str(doTrain))
        print(Fore.BLUE + '                 doTest = ' + str(doTest))
        return tag, jetCollection, analysis, channel, preselectionCuts, backgroundString, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, mass, doTrain, doTest, loop

### Reading from the configuration file
import configparser, ast
import shutil

def ReadConfigSaveToPkl(tag, jetCollection):
    configurationFile = 'Configuration_' + jetCollection + '_' + tag + '.ini'
    config = configparser.ConfigParser()
    config.read(configurationFile)
    ntuplePath = config.get('config', 'ntuplePath')
    inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
    dfPath = config.get('config', 'dfPath')
    dfPath += tag + '/' + jetCollection + '/'
    print (format('Output directory: ' + Fore.GREEN + dfPath), checkCreateDir(dfPath))
    shutil.copyfile(configurationFile, dfPath + configurationFile)
    return ntuplePath, inputFiles, dfPath

def ReadConfig(tag, analysis, jetCollection):
    configurationFile = 'Configuration_' + jetCollection + '_' + tag + '.ini'
    config = configparser.ConfigParser()
    config.read(configurationFile)
    inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
    ntuplePath = config.get('config', 'ntuplePath')
    rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
    signalsList = ast.literal_eval(config.get('config', 'signals'))
    backgroundsList = ast.literal_eval(config.get('config', 'backgrounds'))
    dfPath = config.get('config', 'dfPath')
    dfPath += tag + '/' + jetCollection + '/'
    if analysis == 'merged':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesMerged'))
        variablesToSave = ast.literal_eval(config.get('config', 'variablesToSaveMerged'))
    elif analysis == 'resolved':
        InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolved'))
        variablesToSave = ast.literal_eval(config.get('config', 'variablesToSaveResolved'))
    if sys.argv[0] == fileName2:
        return inputFiles, rootBranchSubSample, InputFeatures, dfPath, variablesToSave, backgroundsList
    if sys.argv[0] == fileNamePlots:
        return dfPath, InputFeatures
    if sys.argv[0] == fileName3:
        return dfPath, InputFeatures, signalsList, backgroundsList
    if sys.argv[0] == fileName4 or sys.argv[0] == fileName5 or sys.argv[0] == fileName6:
        return ntuplePath, dfPath, InputFeatures

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

def LoadData(dfPath, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures):
    fileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + str(signal) + '_' + background + '_' + str(trainingFraction) + 't'
    data_Train = pd.read_pickle(dfPath + '/data_train_' + fileCommonName + '.pkl')
    data_Test = pd.read_pickle(dfPath + '/data_test_' + fileCommonName + '.pkl')
    #data_Train_unscaled = pd.read_pickle(dfPath + '/data_train_unscaled_' + fileCommonName + '.pkl')
    w_train = data_Train['train_weight'].values
    #X_Train_unscaled =  data_Train_unscaled[InputFeatures]
    m_Test_unscaled = pd.read_pickle(dfPath + '/m_test_unscaled_' + fileCommonName + '.pkl').values
    #m_Train_unscaled = data_Train_unscaled['mass'].values
    m_Train_unscaled = pd.read_pickle(dfPath + '/m_train_unscaled_' + fileCommonName + '.pkl').values
    #return data_Train, data_Test, X_Train_unscaled, m_Train_unscaled, m_Test_unscaled, w_train
    return data_Train, data_Test, m_Train_unscaled, m_Test_unscaled, w_train

### Writing in the log file
def WriteLogFile(tag, ntuplePath, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, InputFeatures, dfPath):
    logString = 'CxAOD tag: ' + tag + '\nntuple path: ' + ntuplePath + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nInputFeatures: ' + str(InputFeatures) + '\ndfPath: ' + dfPath# + '\nNumber of train events: ' + str(len(data_train)) + ' (' + str(len(data_train_signal)) + ' signal and ' + str(len(data_train_bkg)) + ' background)' + '\nNumber of test events: ' + str(len(data_test)) + ' (' + str(len(data_test_signal)) + ' signal and ' + str(len(data_test_bkg)) + ' background)'
    return logString

def SelectEvents(dataFrame, channel, analysis, preselectionCuts):
    ### Selecting events according to type of analysis and channel
    selectionMergedGGF = 'Pass_MergHP_GGF_ZZ_Tag_SR == True or Pass_MergHP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_WZ_SR == True or Pass_MergLP_GGF_ZZ_Tag_SR == True or Pass_MergLP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_ZZ_Tag_ZCR == True or Pass_MergHP_GGF_WZ_ZCR == True or Pass_MergHP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_ZZ_Tag_ZCR == True or Pass_MergLP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_WZ_SR == True or Pass_MergLP_GGF_WZ_ZCR == True'
    selectionMergedGGFZZLPuntagSR = 'Pass_MergLP_GGF_ZZ_Untag_SR == True and Pass_MergHP_GGF_ZZ_Tag_SR == False and Pass_MergHP_GGF_ZZ_Untag_SR == False and Pass_MergHP_GGF_WZ_SR == False and Pass_MergLP_GGF_ZZ_Tag_SR == False and Pass_MergHP_GGF_ZZ_Tag_ZCR == False and Pass_MergHP_GGF_WZ_ZCR == False and Pass_MergHP_GGF_ZZ_Untag_ZCR == False and Pass_MergLP_GGF_ZZ_Tag_ZCR == False and Pass_MergLP_GGF_ZZ_Untag_ZCR == False and Pass_MergLP_GGF_WZ_SR == False and Pass_MergLP_GGF_WZ_ZCR == False'
    selectionMergedVBF = 'Pass_MergHP_VBF_WZ_SR == True or Pass_MergHP_VBF_ZZ_SR == True or Pass_MergHP_VBF_WZ_ZCR == True or Pass_MergHP_VBF_ZZ_ZCR == True or Pass_MergLP_VBF_WZ_SR == True or Pass_MergLP_VBF_ZZ_SR == True or Pass_MergLP_VBF_WZ_ZCR == True or Pass_MergLP_VBF_ZZ_ZCR == True'
    selectionResolvedGGF = 'Pass_Res_GGF_WZ_SR == True or Pass_Res_GGF_WZ_ZCR == True or Pass_Res_GGF_ZZ_Tag_SR == True or Pass_Res_GGF_ZZ_Untag_SR == True or Pass_Res_GGF_ZZ_Tag_ZCR == True or Pass_Res_GGF_ZZ_Untag_ZCR == True'
    selectionResolvedVBF = 'Pass_Res_VBF_WZ_SR == True or Pass_Res_VBF_WZ_ZCR == True or Pass_Res_VBF_ZZ_SR == True or Pass_Res_VBF_ZZ_ZCR'
    if channel == 'ggF':
        dataFrame = dataFrame.query('Pass_isVBF == False')
        if analysis == 'merged':
            selection = selectionMergedGGF
            #selection = selectionMergedGGFZZLPuntagSR            
        elif analysis == 'resolved':
            selection = selectionResolvedGGF
    elif channel == 'VBF':
        dataFrame = dataFrame.query('Pass_isVBF == True')
        if analysis == 'merged':
            selection = selectionMergedVBF
        elif analysis == 'resolved':
            selection = selectionResolvedVBF
    dataFrame = dataFrame.query(selection)

    ### Applying preselection cuts (if any)
    if preselectionCuts != 'none':
        dataFrame = dataFrame.query(preselectionCuts)

    return dataFrame

### Selecting signal events according to their mass and type of analysis
def CutMasses(dataFrame, analysis):
    if analysis == 'merged':
        dataFrame = dataFrame.query('mass >= 500')
    elif analysis == 'resolved':
        dataFrame = dataFrame.query('mass <= 1500')
    return dataFrame

### Shuffling dataframe
import sklearn.utils

def ShufflingData(dataFrame):
    dataFrame = sklearn.utils.shuffle(dataFrame, random_state = 123)
    #dataFrame = dataFrame.reset_index(drop = True)
    return dataFrame

### Drawing histograms of each variables in the dataframe divided by class
import seaborn
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams.update({'font.size': 16})

def integral(y,x,bins):
    x_min=x
    s=0
    for i in np.where(bins>x)[0][:-1]:
        s=s+y[i]*(bins[i+1]-bins[i])
    return s

def DrawVariablesHisto(dataFrame, InputFeatures, outputDir, outputFileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts):
    ### Replacing '0' with 'Background' and '1' with 'Signal' in the 'isOrigin' column
    dataFrame['isSignal'].replace(to_replace = [0, 1], value = ['Background', 'Signal'], inplace = True)
    featureLogX = []#'fatjet_D2', 'fatjet_m', 'fatjet_pt', 'lep1_pt', 'lep2_pt', 'Zcand_pt']
    for feature in dataFrame.columns:
        #if feature == 'train_weight' or feature == 'weight' or feature == 'isSignal':
        if feature not in InputFeatures and feature != 'origin':
            continue
        statType = 'probability'
        hueType = dataFrame['isSignal']
        legendBool = True
        if feature == 'origin':
            statType = 'count'
            hueType = dataFrame['origin']
            legendBool = False
        if sys.argv[0] == fileName2:
            ax = seaborn.histplot(data = dataFrame[feature], x = dataFrame[feature], hue = hueType, common_norm = False, stat = statType, legend = legendBool)#, multiple = 'stack')
        elif sys.argv[0] == fileName3:
            contents, bins, _ = plt.hist(dataFrame[feature], weights = dataFrame['train_weight'], bins = 100)
        labelDict = {}
        labelDict['lep1_E'] = 'lep1 E [GeV]'
        labelDict['lep1_m'] = 'lep1 m [GeV]'
        labelDict['lep1_pt'] = r'lep1 p$_T$ [GeV]'
        labelDict['lep1_eta'] = r'lep1 $\eta$'        
        labelDict['lep1_phi'] = r'lep1 $\phi$'
        labelDict['lep2_E'] = 'lep2 E [GeV]'
        labelDict['lep2_m'] = 'lep2 m [GeV]'
        labelDict['lep2_pt'] = r'lep2 p$_t$ [GeV]'
        labelDict['lep2_eta'] = r'lep2 $\eta$'        
        labelDict['lep2_phi'] = r'lep2 $\phi$'
        labelDict['fatjet_m'] = 'fatjet m [GeV]'
        labelDict['fatjet_pt'] = r'fatjet p$_t$ [GeV]'
        labelDict['fatjet_eta'] = r'fatjet $\eta$'        
        labelDict['fatjet_phi'] = r'fatjet $\phi$'
        labelDict['fatjet_D2'] = r'fatjet D$_2$'
        labelDict['Zcand_m'] = 'Zcand m [GeV]'
        labelDict['Zcand_pt'] = r'Zcand p$_t$ [GeV]'
        labelDict['X_boosted_m'] = 'X_boosted m [GeV]'
        labelDict['mass'] = 'mass [GeV]'
        labelDict['weight'] = 'weight'
        labelDict['isSignal'] = 'isSignal'
        labelDict['origin'] = 'origin'
        '''
        if feature in featureLogX:
            ax.set_xscale('log')
        '''
        legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
        if (preselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + preselectionCuts
        #plt.figtext(0.35, 0.7, legendText, wrap = True, horizontalalignment = 'left')
        #plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
        #plt.subplots_adjust(left = 0.1, right = 0.75)
        plt.xlabel(labelDict[feature])
        if sys.argv[0] == fileName3:
            plt.ylabel('Weighted counts')
        if feature == 'origin':
            plt.yscale('log')
        pltName = '/Histo_' + feature + '_' + outputFileCommonName + '.png'
        plt.tight_layout()
        plt.savefig(outputDir + pltName)
        print(Fore.GREEN + 'Saved ' + outputDir + pltName)
        plt.clf()
    dataFrame['isSignal'].replace(to_replace = ['Background', 'Signal'], value = [0, 1], inplace = True)
    #plt.subplots_adjust(left = 0.15, right = 0.95)
    return

### Computing train weight
def ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts):
    numbersDict = {}
    for signalMass in massesSignalList:
        ### Number of signals with each mass value
        numbersDict[signalMass] = dataSetSignal[dataSetSignal['mass'] == signalMass].shape[0]
        print(Fore.BLUE + 'Number of events with mass ' + str(signalMass) + ': ' + str(numbersDict[signalMass]))

    ### Minimum number of signals with the same mass
    minNumber = min(numbersDict.values())
    ### New column in the signal dataframe for train_weight
    dataSetSignal = dataSetSignal.assign(train_weight = 0)
    for signalMass in massesSignalList:
        dataSetSignalMass = dataSetSignal[dataSetSignal['mass'] == signalMass]
        ### Sum of MC weights for signal with the same mass
        signalSampleWeight = dataSetSignalMass['weight'].sum()
        ### Scale factor to equalize signal samples with different mass
        scaleFactorDict = minNumber / signalSampleWeight
        ### Train weight = MC weight * scale factor
        dataSetSignal['train_weight'] = np.where(dataSetSignal['mass'] == signalMass, dataSetSignal['weight'] * scaleFactorDict, dataSetSignal['train_weight'])
        ### Filling the bar plot
        plt.bar(signal + ' ' + str(signalMass) + ' GeV', dataSetSignal[dataSetSignal['mass'] == signalMass]['train_weight'].sum(), color = 'blue')
    plt.bar('all ' + signal, dataSetSignal['train_weight'].sum(), color = 'orange')
    ### All signal weight
    signalWeight = dataSetSignal['train_weight'].sum()
    ### Background MC weight
    bkgWeight = dataSetBackground['weight'].sum()
    ### Scale factor to equalize signal/background
    scaleFactor = minNumber *  len(massesSignalList)/ bkgWeight
    ### Creating new column in the background dataframe with the train weight
    dataSetBackground = dataSetBackground.assign(train_weight = dataSetBackground['weight'] * scaleFactor)

    plt.bar('background', dataSetBackground['train_weight'].sum(), color = 'green')
    plt.ylabel('Weighted counts')
    plt.xticks(rotation = 'vertical')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    plt.figtext(0.25, 0.7, legendText, wrap = True, horizontalalignment = 'left')
    plt.tight_layout()
    pltName = outputDir + '/WeightedEvents_' + fileCommonName + '.png'
    plt.savefig(pltName)
    print(Fore.GREEN + 'Saved ' + pltName)
    plt.clf()
    return dataSetSignal, dataSetBackground
    

### Computing weighted median and IQR range
def ComputeStat(dataFrameColumn, trainWeights, perc1, perc2, name):
    contents, bins, _ = plt.hist(dataFrameColumn, weights = trainWeights, bins = 100)
    #if name == 'fatjet_m':
    #    print(bins)
    plt.clf()
    binCenters = bins[:-1] + np.diff(bins)/2
    #if name == 'fatjet_m':
    #    print(binCenters)
    pdf = contents / sum(contents)
    cdf = np.cumsum(pdf)
    plt.plot(binCenters, cdf, label="CDF")
    plt.legend()
    #plt.show()
    plt.savefig(name + '.png')
    plt.clf()
    weightedMedian = np.interp(0.5, cdf, binCenters)
    print(weightedMedian)
    q1 = np.interp(perc1, cdf, binCenters) ###0.25
    q2 = np.interp(perc2, cdf, binCenters) ###0.75
    iqr = q2 - q1
    print('IQR: ' + str(iqr))
    #dataFrameColumnScaled = (dataFrameColumn - weightedMedian) / iqr
    return weightedMedian, iqr

def ComputeStatNew(dataTrain, dataTest, InputFeatures, outputDir):
    sumTrainWeights = np.array(dataTrain['train_weight']).sum()
    halfTrainWeights = sumTrainWeights / 2
    perc1 = sumTrainWeights * 0.25
    perc2 = sumTrainWeights * 0.75
    variablesFileName = outputDir + '/variables.json'
    variablesFile = open(variablesFileName, 'w')
    variablesFile.write('{\n')
    variablesFile.write('  \'inputs\': [\n')
    for feature in InputFeatures:
        cumulativeSum = 0
        print('Scaling ' + feature)
        dataTrain = dataTrain.sort_values(by = [feature])
        for index in range(len(dataTrain)):
            cumulativeSum += dataTrain['train_weight'].iloc[index]
            if cumulativeSum <= perc1:
                perc1index = index
            else:
                break
        for index in range(perc1index, len(dataTrain)):
            cumulativeSum += dataTrain['train_weight'].iloc[index]
            if cumulativeSum <= halfTrainWeights:
                medianIndex = index
            else:
                break
        for index in range(medianIndex, len(dataTrain)):
            cumulativeSum += dataTrain['train_weight'].iloc[index]
            if cumulativeSum <= perc2:
                perc2index = index
            else:
                break
        quartileLeft = dataTrain[feature].iloc[perc1index]
        median = dataTrain[feature].iloc[medianIndex]
        quartileRight = dataTrain[feature].iloc[perc2index]
        iqr = quartileRight - quartileLeft ### InterQuartile Range
        dataTrain[feature] = (dataTrain[feature] - median) / iqr
        dataTest[feature] = (dataTest[feature] - median) / iqr
        variablesFile.write('    {\n')
        variablesFile.write('      \'name\': \'%s\',\n' % feature)
        variablesFile.write('      \'offset\': %lf,\n' % median) # EJS 2021-05-27: I have compelling reasons to believe this should be -mu
        variablesFile.write('      \'scale\': %lf\n' % iqr) # EJS 2021-05-27: I have compelling reasons to believe this should be 1/sigma                            
        variablesFile.write('    }')
        if feature != dataTrain.columns[len(dataTrain.columns) - 1]:
            variablesFile.write(',\n')
        else:
            variablesFile.write('\n')
    variablesFile.write('  ],\n')
    variablesFile.write('  \'class_labels\': [\'BinaryClassificationOutputName\']\n')
    variablesFile.write('}\n')
    print(Fore.GREEN + 'Saved variables in ' + variablesFileName)
    return dataTrain, dataTest

### Building the (P)DNN
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation
import tensorflow as tf
from sklearn.metrics import log_loss

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
    Loss = 'binary_crossentropy'
    Metrics = ['accuracy']
    learningRate = 0.0003 #0.001
    Optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate) #Adam
    return model, Loss, Metrics, learningRate, Optimizer

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
'''    
def SaveModel(model, X_input, outputDir):
    SaveArchAndWeights(model, outputDir)
    SaveVariables(outputDir, X_input)
    SaveFeatureScaling(outputDir, X_input)
'''

def SaveModel(model, outputDir, NN):
    SaveArchAndWeights(model, outputDir)
    variablesFileName = '/variables.json'
    previousDir = outputDir.replace(NN, '')
    print(previousDir)
    shutil.copyfile(previousDir + variablesFileName, outputDir + variablesFileName)

'''
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams.update({'font.size': 16})
'''
### Evaluating the (P)DNN performance
def EvaluatePerformance(model, X_test, y_test, batchSize):
    perf = model.evaluate(X_test, y_test, batch_size = batchSize)
    testLoss = perf[0]
    testAccuracy = perf[1]
    return testLoss, testAccuracy

### Prediction on train and test sample (for DNN)
def PredictionTrainTest(model, X_test, X_train, batchSize):
    yhat_test = model.predict(X_test, batch_size = batchSize)
    yhat_train = model.predict(X_train, batch_size = batchSize)
    return yhat_test, yhat_train
    
### Prediction on signal and background separately (for PDNN)
def PredictionSigBkg(model, X_train_signal, X_train_bkg, X_test_signal, X_test_bkg, batchSize):
    yhat_train_signal = model.predict(X_train_signal, batch_size = batchSize)
    yhat_train_bkg = model.predict(X_train_bkg, batch_size = batchSize)
    yhat_test_signal = model.predict(X_test_signal, batch_size = batchSize)
    yhat_test_bkg = model.predict(X_test_bkg, batch_size = batchSize)
    return yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg

### Drawing correlation matrix
def DrawCorrelationMatrix(dataFrame, InputFeatures, outputDir, outputFileCommonName, analysis, channel, signal, bkg):
    fig, ax1 = plt.subplots(figsize = (10, 10))
    plt.set_cmap('bwr')
    im = ax1.matshow((dataFrame[InputFeatures].astype(float)).corr(), vmin = -1, vmax = 1)
    plt.colorbar(im, ax = ax1)
    plt.xticks(range(len(InputFeatures)), InputFeatures, rotation = 'vertical')
    plt.yticks(range(len(InputFeatures)), InputFeatures)
    '''
    for feature1 in range(len(InputFeatures)): ### This is really slow, perform only if needed
        for feature2 in range(len(InputFeatures)):
            ax1.text(feature2, feature1, "%.2f" % (dataFrame[InputFeatures].astype(float)).corr().at[InputFeatures[feature2], InputFeatures[feature1]], ha = 'center', va = 'center', color = 'r', fontsize = 6)
    '''
    plt.title('Correlation matrix (' + analysis + ' ' + channel + ' ' + signal + ' ' + bkg + ')')
    CorrelationMatrixName = outputDir + '/CorrelationMatrix_' + outputFileCommonName + '.png'
    plt.savefig(CorrelationMatrixName)
    print(Fore.GREEN + 'Saved ' + CorrelationMatrixName)
    plt.clf()

### Drawing Accuracy
def DrawAccuracy(modelMetricsHistory, testAccuracy, patienceValue, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName, mass = 0):
    plt.plot(modelMetricsHistory.history['accuracy'], label = 'Training')
    lines = plt.plot(modelMetricsHistory.history['val_accuracy'], label = 'Validation')
    xvalues = lines[0].get_xdata()
    if testAccuracy != None:
        plt.scatter([xvalues[len(xvalues) - 1 - patienceValue]], [testAccuracy], label = 'Test', color = 'green')
    emptyPlot, = plt.plot([0, 0], [1, 1], color = 'white')
    titleAccuracy = NN + ' model accuracy'
    if NN == 'DNN':
        outputFileCommonName += '_' + str(int(mass))
        if mass >= 1000:
            titleAccuracy += ' (mass: ' + str(float(mass / 1000)) + ' TeV)'
        else:
            titleAccuracy += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleAccuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    #plt.legend()
    #legend1 = plt.legend(['Training', 'Validation'], loc = 'lower right')
    legend1 = plt.legend(loc = 'center right')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    #legendText += '\nTest accuracy: ' + str(round(testAccuracy, 2))
    #plt.figtext(0.5, 0.3, legendText, wrap = True, horizontalalignment = 'left')
    #plt.legend(legendText)
    legend2 = plt.legend([emptyPlot], [legendText], frameon = False)
    plt.gca().add_artist(legend1)
    #plt.figtext(0.69, 0.28, 'Test accuracy: ' + str(round(testAccuracy, 2)), wrap = True, horizontalalignment = 'left')#, fontsize = 10)
    AccuracyPltName = outputDir + '/Accuracy_' + outputFileCommonName + '.png'
    plt.savefig(AccuracyPltName)
    print(Fore.GREEN + 'Saved ' + AccuracyPltName)
    plt.clf()
        
### Drawing Loss
def DrawLoss(modelMetricsHistory, testLoss, patienceValue, outputDir, NN, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName, mass = 0):
    plt.plot(modelMetricsHistory.history['loss'], label = 'Training')
    lines = plt.plot(modelMetricsHistory.history['val_loss'], label = 'Validation')
    xvalues = lines[0].get_xdata()
    #print(yvalues[len(yvalues) - 1])    
    if testLoss != None:
        plt.scatter([xvalues[len(xvalues) - 1 - patienceValue]], [testLoss], label = 'Test', color = 'green')
    #emptyPlot, = plt.plot([0, 0], [1, 1], color = 'white')
    titleLoss = NN + ' model loss'
    if NN == 'DNN':
        outputFileCommonName += '_' + str(int(mass))
        if mass >= 1000:
            titleLoss += ' (mass: ' + str(float(mass / 1000)) + ' TeV)'
        else:
            titleLoss += ' (mass: ' + str(int(mass)) + ' GeV)'
    plt.title(titleLoss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Training', 'Validation'], loc = 'upper right')
    legend1 = plt.legend(loc = 'upper center')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\npreselection cuts: ' + PreselectionCuts + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
    if (PreselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + PreselectionCuts
    #legendText += '\nTest loss: ' + str(round(testLoss, 2))
    plt.figtext(0.4, 0.4, legendText, wrap = True, horizontalalignment = 'left')
    #plt.figtext(0.7, 0.7, 'Test loss: ' + str(round(testLoss, 2)), wrap = True, horizontalalignment = 'center')#, fontsize = 10)
    #legend2 = plt.legend([emptyPlot], [legendText], frameon = False, loc = 'center right')
    #plt.gca().add_artist(legend1)
    LossPltName = outputDir + '/Loss_' + outputFileCommonName + '.png'
    plt.savefig(LossPltName)
    print(Fore.GREEN + 'Saved ' + LossPltName)
    plt.clf()

from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
'''
def DrawROC(fpr, tpr, AUC, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background):
    plt.plot(fpr,  tpr, color = 'darkorange', lw = 2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    titleROC = 'ROC curves (mass: ' + str(int(unscaledMass)) + ' GeV)'
    plt.title(titleROC)
    plt.figtext(0.7, 0.25, 'AUC: ' + str(round(AUC, 2)), wrap = True, horizontalalignment = 'center')
    ROCPltName = outputDir + '/oldROC.png'
    plt.savefig(ROCPltName)
    print('Saved ' + ROCPltName)
    plt.clf()
'''

### Drawing ROC and background rejection vs efficiency
def DrawROCbkgRejectionScores(fpr, tpr, AUC, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, bkgNumber, yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg):

    ### ROC
    emptyPlot, = plt.plot(fpr[0], tpr[0], color = 'white')
    plt.plot(fpr, tpr, color = 'darkorange', label = 'AUC: ' + str(round(AUC, 2)), lw = 2)
    legend1 = plt.legend(handlelength = 0, handletextpad = 0, loc = 'center right')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if unscaledMass >= 1000:
        titleScores = NN + ' scores (mass: ' + str(float(unscaledMass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
        titleROC = NN + ' ROC curve (mass: ' + str(float(unscaledMass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
        BkgRejTitle = NN + ' rejection curve (mass: ' + str(float(unscaledMass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
    else:
        titleScores = NN + ' scores (mass: ' + str(int(unscaledMass)) + ' GeV)'#, bkg: ' + background + ')'
        titleROC = NN + ' ROC curve (mass: ' + str(int(unscaledMass)) + ' GeV)'#, bkg: ' + background + ')'
        BkgRejTitle = NN + ' rejection curve (mass: ' + str(int(unscaledMass)) + ' GeV)'#, bkg: ' + background + ')'
    plt.title(titleROC)

    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    legend2 = plt.legend([emptyPlot], [legendText], loc = 'lower right', handlelength = 0, handletextpad = 0)
    for item in legend2.legendHandles:
        item.set_visible(False)

    plt.gca().add_artist(legend1)
    ROCPltName = outputDir + '/ROC_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
    plt.savefig(ROCPltName)#, bbox_inches = 'tight')
    print(Fore.GREEN + 'Saved ' + ROCPltName)
    plt.clf()
    
    ### Scores
    Nbins = 1000
    plt.hist(yhat_train_signal, bins = Nbins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal train'], density = True)
    y_signal, bins_1, _ = plt.hist(yhat_test_signal, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal test'], density = True)
    plt.hist(yhat_train_bkg, bins = Nbins, histtype = 'step', lw = 2, color = 'red', label = [r'Background train'], density = True)
    y_bkg, bins_0, _ = plt.hist(yhat_test_bkg, bins = Nbins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background test'], density = True)
    plt.ylabel('Norm. entries')
    plt.xlabel('Score')
    plt.yscale('log')
    plt.title(titleScores)
    plt.legend(loc = 'upper center')
    ScoresPltName = outputDir + '/Scores_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
    plt.savefig(ScoresPltName)
    print(Fore.GREEN + 'Saved ' + ScoresPltName)
    plt.clf()

    ### Background rejection vs efficiency
    tprCut = tpr[tpr > 0.85]
    fprCut = fpr[tpr > 0.85]
    fprCutInverse = 1 / fprCut
    plt.plot(tprCut, fprCutInverse)
    emptyPlot, = plt.plot(tprCut[0], fprCutInverse[0] + 30, color = 'white')
  
    WP = [0.90, 0.94, 0.97, 0.99]
    bkgRejections = np.array([])
    #print(Fore.BLUE + 'Mass: ' + str(unscaledMass), ', background rejection at WP = 0.90: ' + str(bkgRej90))

    for i in range(0, len(WP)):
        bkgRejections = np.append(bkgRejections, np.interp(WP[i], tprCut, fprCutInverse))
        plt.axvline(x = WP[i], color = 'Red', linestyle = 'dashed', label = 'Bkg Rejection @ ' + str(WP[i]) + ' WP: ' + str(round(bkgRejections[i], 1)))

    print(format(Fore.BLUE + 'Working points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(bkgRejections)))
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection')
    plt.yscale('log')    
    plt.title(BkgRejTitle)
    legend1 = plt.legend()
    legend2 = plt.legend([emptyPlot], [legendText], loc = 'center left', handlelength = 0, handletextpad = 0)
    for item in legend2.legendHandles:
        item.set_visible(False)
    plt.gca().add_artist(legend1)
    EffPltName = outputDir + '/BkgRejection_' + outputFileCommonName + '_' + str(unscaledMass) + '.png'
    plt.savefig(EffPltName)#, bbox_inches = 'tight')
    print(Fore.GREEN + 'Saved ' + EffPltName)
    plt.clf()

    return WP, bkgRejections


def integral(y,x,bins):
    x_min=x
    s=0
    for i in np.where(bins>x)[0][:-1]:
        s=s+y[i]*(bins[i+1]-bins[i])
    return s

### Drawing scores, ROC and efficiency
def DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass, jetCollection, analysis, channel, PreselectionCuts, signal, bkg, outputFileCommonName):

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
            titleScores = NN + ' scores (mass: ' + str(float(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            titleScores = NN + ' scores (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(titleScores)
        plt.legend(loc = 'upper center')
        legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        #plt.figtext(0.35, 0.45, legendText, wrap = True, horizontalalignment = 'left')
        #ScoresPltName = outputDir + '/Scores_' + bkg + '.png'
        ScoresPltName = outputDir + '/Scores_' + outputFileCommonName + '_' + str(mass) + '.png'
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
            titleROC = NN + ' ROC curve (mass: ' + str(float(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            titleROC = NN + ' ROC curve (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(titleROC)
        legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(bkg)
        if (PreselectionCuts != 'none'):
            legendText += '\npreselection cuts: ' + PreselectionCuts
        plt.figtext(0.4, 0.25, legendText, wrap = True, horizontalalignment = 'left')
        plt.figtext(0.4, 0.2, 'AUC: ' + str(round(Area, 2)), wrap = True, horizontalalignment = 'center')
        #ROCPltName = outputDir + '/ROC_' + bkg + '.png'
        ROCPltName = outputDir + '/ROC_' + outputFileCommonName + '_' + str(mass) + '.png'
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
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(float(mass / 1000)) + ' TeV, bkg: ' + bkg + ')'
        else:
            BkgRejTitle = NN + ' rejection curve (mass: ' + str(int(mass)) + ' GeV, bkg: ' + bkg + ')'
        plt.title(BkgRejTitle)
        plt.legend()
        #EffPltName = outputDir + '/BkgRejection_' + bkg +'.png'
        EffPltName = outputDir + '/BkgRejection_' + outputFileCommonName + '_' + str(mass) + '.png'
        plt.savefig(EffPltName)
        print(Fore.GREEN + 'Saved ' + EffPltName)
        plt.clf()
    return Area, WP, WP_rej 

def DrawRejectionVsMass(massVec, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName):

    emptyPlot, = plt.plot(massVec[0], bkgRej90[0], color = 'white')
    plt.plot(massVec, bkgRej90, color = 'blue', label = 'WP: ' + str(WP[0]), marker = 'o', mec = 'blue')
    plt.plot(massVec, bkgRej94, color = 'orange', label = 'WP: ' + str(WP[1]), marker = 'o', mec = 'orange')
    plt.plot(massVec, bkgRej97, color = 'green', label = 'WP: ' + str(WP[2]), marker = 'o', mec = 'green')
    plt.plot(massVec, bkgRej99, color = 'red', label = 'WP: ' + str(WP[3]), marker = 'o', mec = 'red')
    plt.yscale('log')
    legend1 = plt.legend()
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Background rejection')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    '''
    legend2 = plt.legend([emptyPlot], [legendText], loc = 'lower left', handlelength = 0, handletextpad = 0)
    for item in legend2.legendHandles:
        item.set_visible(False)
    plt.gca().add_artist(legend1)#, bbox_to_anchor = (1.05, 0.6))
    '''
    plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
    plt.subplots_adjust(left = 0.15, right = 0.75)
    pltName = outputDir + '/BkgRejectionVsMass_' + outputFileCommonName + '.png'
    plt.savefig(pltName)#, bbox_inches = 'tight')
    print(Fore.GREEN + 'Saved ' + pltName)
    plt.subplots_adjust(left = 0.15, right = 0.95)

def DrawRejectionVsStat(massVec, fracTrain, WP, bkgRej90Dict, bkgRej94Dict, bkgRej97Dict, bkgRej99Dict, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName):
    emptyPlot, = plt.plot(massVec[0], bkgRej90Dict[fracTrain[0]], color = 'white')
    linestyles = ['loosely dotted', 'dotted', 'loosely dashed', 'solid']
    for i in range(len(fracTrain)):
        frac = fracTrain[i]
        print(bkgRej90Dict[frac])
        plt.plot(massVec, np.array(bkgRej90Dict[frac]), color = 'blue', label = 'WP: ' + str(WP[0]), marker = 'o', mec = 'blue', linestyle = linestyles[i])
        plt.plot(massVec, np.array(bkgRej94Dict[frac]), color = 'orange', label = 'WP: ' + str(WP[1]), marker = 'o', mec = 'orange', linestyle = linestyles[i])
        plt.plot(massVec, np.array(bkgRej97Dict[frac]), color = 'green', label = 'WP: ' + str(WP[2]), marker = 'o', mec = 'green', linestyle = linestyles[i])
        plt.plot(massVec, np.array(bkgRej99Dict[frac]), color = 'red', label = 'WP: ' + str(WP[3]), marker = 'o', mec = 'red', linestyle = linestyles[i])
    plt.yscale('log')
    legend1 = plt.legend()
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Background rejection')
    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    '''
    legend2 = plt.legend([emptyPlot], [legendText], loc = 'lower left', handlelength = 0, handletextpad = 0)
    for item in legend2.legendHandles:
        item.set_visible(False)
    plt.gca().add_artist(legend1)#, bbox_to_anchor = (1.05, 0.6))
    '''
    plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
    plt.subplots_adjust(left = 0.15, right = 0.75)
    pltName = outputDir + '/BkgRejectionVsStat_' + outputFileCommonName + '.png'
    plt.savefig(pltName)#, bbox_inches = 'tight')
    print(Fore.GREEN + 'Saved ' + pltName)
    plt.subplots_adjust(left = 0.15, right = 0.95)


from sklearn.metrics import confusion_matrix
import itertools

def DrawCM(yhat_test, y_test, outputDir, mass, background, outputFileCommonName, jetCollection, analysis, channel, preselectionCuts, signal):
    yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test])
    cm = confusion_matrix(y_test, yResult_test_cls, normalize = 'true')
    TNR, FPR, FNR, TPR = cm.ravel()
    print(format(Fore.BLUE + 'TNR: '  + str(TNR) + ', FPR: ' + str(FPR) + ', FNR: ' + str(FNR) + ', TPR: ' + str(TPR)))
    classes = ['Background', 'Signal']
    np.set_printoptions(precision = 2)
    cmap = plt.cm.Oranges
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    if mass >= 1000:
        titleCM = 'Confusion matrix (mass: ' + str(float(mass / 1000)) + ' TeV)'#, bkg: ' + background + ')'
    else:
        titleCM = 'Confusion matrix (mass: ' + str(int(mass)) + ' GeV)'#, bkg: ' + background + ')'
    plt.title(titleCM)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation = 90)
    #thresh = cm.max() / 2.
    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    CMPltName = outputDir + '/ConfusionMatrix_' + outputFileCommonName + '_' + str(mass) + '.png'

    legendText = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\nsignal: ' + signal + '\nbackground: ' + str(background)
    if (preselectionCuts != 'none'):
        legendText += '\npreselection cuts: ' + preselectionCuts
    plt.figtext(0.77, 0.45, legendText, wrap = True, horizontalalignment = 'left')
    plt.subplots_adjust(left = 0.1, right = 0.75)
    plt.savefig(CMPltName)#, bbox_inches = 'tight')
    print(Fore.GREEN + 'Saved ' + CMPltName)
    plt.subplots_adjust(left = 0.15, right = 0.95)
    plt.clf()
    return TNR, FPR, FNR, TPR

def weightEventsOld(origin_train):
    originsList = np.array(list(set(list(origin_train))))
    originsNumber = np.array([])
    for origin in originsList:
        originsNumber = np.append(originsNumber, list(origin_train).count(origin))
    #print(originsNumber)
    minNumber = min(originsNumber)
    #weights = minNumber / originsNumber
    weights = minNumber / ( 2 * originsNumber)
    weights = np.where(weights == 0.5, 1, weights)
    #print(weights)
    w_origin_train = origin_train.copy()
    for origin in originsList:
        w_origin_train = np.where(w_origin_train == str(origin), weights[np.where(originsList == origin)], w_origin_train)
    w_origin_train = np.asarray(w_origin_train).astype(np.float32)
    return w_origin_train, originsList, originsNumber, weights


def weightEvents(origin_train, signal):
    DictNumbers = {}
    originsList = np.array(list(set(list(origin_train))))
    for origin in originsList:
        DictNumbers[origin] = list(origin_train).count(origin)
    minNumber = min(DictNumbers.values())
    minOrigin = [key for key in DictNumbers if DictNumbers[key] == minNumber]
    DictWeights = {}
    if signal in minOrigin:
        for origin in originsList:
            if origin == signal:
                DictWeights[origin] = 1
            else:
                DictWeights[origin] = minNumber / ((len(originsList) - 1) * DictNumbers[origin]) ### DictNumbers[signal]
    elif DictNumbers[signal] > ((len(originsList) - 1) * minNumber):
        for origin in originsList:
            if origin == signal:
                DictWeights[origin] = (len(originsList) - 1) * minNumber / DictNumbers[origin]
            elif origin in minOrigin:
                DictWeights[origin] = 1
            else:
                DictWeights[origin] = minNumber / DictNumbers[origin]
    else:
        print(Fore.RED + 'No weights defined for this statistic')
    w_origin_train = origin_train.copy()
    for origin in originsList:
        w_origin_train = np.where(w_origin_train == origin, DictWeights[origin], w_origin_train)
    w_origin_train = np.asarray(w_origin_train).astype(np.float32)
    return w_origin_train, originsList, DictNumbers, DictWeights

'''
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
'''
