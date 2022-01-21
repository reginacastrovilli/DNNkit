from Functions import ReadArgParser, checkCreateDir, ReadConfig, SaveFeatureScaling, DrawCorrelationMatrix, DrawVariablesHisto, weightEvents
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
drawPlot = False

for signal in signalsList:
    ### Selecting only the request signal
    if signal not in testSignal:
        continue
    foundSignal += 1

    ### Loading input file
    inputOutputDir = dfPath + analysis + '/' + channel + '/' + signal + '/' + background
    fileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background
    data = pd.read_pickle(inputOutputDir + '/MixData_PD_' + fileCommonName + '.pkl') 

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
        print(Fore.BLUE + 'Number of ' + origin + ' events: ' + str(dictOrigin[origin]))#str(list(data_set['origin']).count(origin)))
        #print(Fore.BLUE + 'Number of ' + origin + ' events: ' + str(list(data_set['origin']).count(origin)))

    if(drawPlot):
        ### Plotting histograms of each variables divided by class and the correlation matrix
        DrawVariablesHisto(data_set, inputOutputDir, fileCommonName, analysis, channel, signal, background)
        #DrawCorrelationMatrix(data_set, InputFeatures, inputOutputDir, fileCommonName, analysis, channel, signal, background)

    ### Splitting data into train and test set
    data_train, data_test = train_test_split(data_set, train_size = trainingFraction)

    ### Saving unscaled data
    #fileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + '_' + str(trainingFraction) + 't'
    fileCommonName += '_' + str(trainingFraction) + 't'
    dataTrainUnscaledName = inputOutputDir + '/data_train_unscaled_' + fileCommonName + '.pkl' 
    data_train.to_pickle(dataTrainUnscaledName)
    print(Fore.GREEN + 'Saved ' + dataTrainUnscaledName)
    m_test_unscaled = data_test['mass']
    mTestUnscaledName = inputOutputDir + '/m_test_unscaled_' + fileCommonName + '.pkl' 
    m_test_unscaled.to_pickle(mTestUnscaledName)
    print(Fore.GREEN + 'Saved ' + mTestUnscaledName)

    ### Saving list of columns names
    columnsNames = data_train.columns
    origin_train = np.array(data_train['origin'].values)
    weights, _, _, _ = weightEvents(origin_train, str(signal))

    ### Scaling InputFeatures of train and test set
    data_train_copy = data_train.copy()
    for column in InputFeatures:
        data_train_copy[column] *= weights
        median = data_train_copy[column].median()
        q75, q25 = np.percentile(data_train_copy[column], [75, 25])
        iqr = q75 - q25
        data_train[column] = (data_train[column] - median) / iqr
        data_test[column] = (data_test[column] - median) / iqr

    '''
    scaler_train = ct.fit(data_train, None)
    data_train = scaler_train.transform(data_train)
    data_test = scaler_train.transform(data_test)
    print(scaler_train.mean_)
    print(scaler_train.var_)
    seaborn.histplot(data = data_train['lep2_eta'], x = data_train['lep2_eta'], hue = dataFrame['origin'], common_norm = False, stat = statType, legend = True)
    plt.show()
    exit()
    '''
    ### Converting numpy arrays into pandas dataframes
    #data_train = pd.DataFrame(data_train, columns = columnsNames)
    #data_test = pd.DataFrame(data_test, columns = columnsNames)
    
    ### Saving scaled dataframes
    dataTrainName = inputOutputDir + '/data_train_' + fileCommonName + '.pkl'
    data_train.to_pickle(dataTrainName)
    print(Fore.GREEN + 'Saved ' + dataTrainName)
    dataTestName = inputOutputDir + '/data_test_' + fileCommonName + '.pkl'
    data_test.to_pickle(dataTestName)
    print(Fore.GREEN + 'Saved ' + dataTestName)

if foundSignal == 0:
    print(Fore.RED + 'Requested signal not found')
