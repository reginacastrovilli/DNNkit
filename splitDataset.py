from Functions import ReadArgParser, checkCreateDir, ReadConfig, SaveFeatureScaling, DrawVariablesHisto, ShufflingData, ComputeTrainWeights, ComputeStat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

pd.options.mode.chained_assignment = None ### to suppress the SettingWithCopyWarning

from colorama import init, Fore
init(autoreset = True)

### Reading from command line
tag, jetCollection, analysis, channel, preselectionCuts, background, testSignal, trainingFraction = ReadArgParser()

### Reading from configuration file
dfPath, InputFeatures, signalsList, backgroundsList = ReadConfig(tag, analysis, jetCollection)

### Creating the list of signals to take
if testSignal == 'all':
    testSignal = signalsList.copy()
else:
    testSignal = list(testSignal.split('_'))

foundSignal = 0
drawPlots = False

for signal in signalsList:
    ### Selecting only the request signal
    if signal not in testSignal:
        continue
    foundSignal += 1

    ### Loading input file
    inputDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background
    fileCommonName = tag + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
    data = pd.read_pickle(inputDir + '/MixData_' + fileCommonName + '.pkl') 
    
    ### If not already existing, creating output directory
    outputDir = inputDir
    checkCreateDir(outputDir) 
    fileCommonName += '_' + str(trainingFraction) + 't'

    ### Creating the list of backgrounds and signal processes to select
    if background == 'all':
        inputOrigin = backgroundsList.copy()
    else:
        inputOrigin = list(background.split('_'))
    inputOrigin.append(signal)

    ### Selecting events according to their origin 
    data_set = data[data['origin'].isin(inputOrigin)]

    nEvents = round(data_set.shape[0] * 2 / 4)
    #data_set = data_set[:nEvents]

    dictOrigin = {}
    for origin in inputOrigin:
        dictOrigin[origin] = list(data_set['origin']).count(origin)
        print(Fore.BLUE + 'Number of ' + origin + ' events: ' + str(dictOrigin[origin]))
        
    ### Dividing signal from background
    dataSetSignal = data_set[data_set['origin'] == signal]
    dataSetBackground = data_set[data_set['origin'] != signal]
    print(Fore.BLUE + '---> Number of background events: ' + str(dataSetBackground.shape[0]))
    print(Fore.BLUE + '---> Number of signal events: ' + str(dataSetSignal.shape[0]))
    massesSignalList = sorted(list(set(list(dataSetSignal['mass']))))
    print(Fore.BLUE + 'Masses in the signal sample: ' + str(massesSignalList) + ' (' + str(len(massesSignalList)) + ')')

    ### Creating new column in the dataframes with train weight
    dataSetSignal, dataSetBackground = ComputeTrainWeights(dataSetSignal, dataSetBackground, massesSignalList, outputDir, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts)

    ### Concatening signal and background dataframes
    dataFrame = pd.concat([dataSetSignal, dataSetBackground], ignore_index = True)

    ### Shuffling the dataframe
    dataFrame = ShufflingData(dataFrame)
    
    ### Splitting data into train and test set
    data_train, data_test = train_test_split(dataFrame, train_size = trainingFraction)

    if drawPlots:
        trainHistogramsPath = outputDir + '/trainUnscaledHistograms'
        print (format('Output directory: ' + Fore.GREEN + trainHistogramsPath), checkCreateDir(trainHistogramsPath))
        DrawVariablesHisto(data_train, InputFeatures, trainHistogramsPath, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts)

    ### Saving unscaled masses
    m_train_unscaled = data_train['mass']
    m_test_unscaled = data_test['mass']
    mTrainUnscaledName = outputDir + '/m_train_unscaled_' + fileCommonName + '.pkl' 
    mTestUnscaledName = outputDir + '/m_test_unscaled_' + fileCommonName + '.pkl' 
    m_train_unscaled.to_pickle(mTrainUnscaledName)
    m_test_unscaled.to_pickle(mTestUnscaledName)
    print(Fore.GREEN + 'Saved ' + mTrainUnscaledName)
    print(Fore.GREEN + 'Saved ' + mTestUnscaledName)

    ### Scaling InputFeatures of train and test set
    data_train, data_test = ComputeStat(data_train, data_test, InputFeatures, outputDir)

    if drawPlots:
        scaledHistogramsPath = outputDir + '/trainScaledHistograms'
        print (format('Output directory: ' + Fore.GREEN + scaledHistogramsPath), checkCreateDir(scaledHistogramsPath))
        DrawVariablesHisto(data_train, InputFeatures, scaledHistogramsPath, fileCommonName, jetCollection, analysis, channel, signal, background, preselectionCuts)

    ### Saving scaled dataframes
    dataTrainName = outputDir + '/data_train_' + fileCommonName + '.pkl'
    data_train.to_pickle(dataTrainName)
    print(Fore.GREEN + 'Saved ' + dataTrainName)
    dataTestName = outputDir + '/data_test_' + fileCommonName + '.pkl'
    data_test.to_pickle(dataTestName)
    print(Fore.GREEN + 'Saved ' + dataTestName)

if foundSignal == 0:
    print(Fore.RED + 'Requested signal not found')
