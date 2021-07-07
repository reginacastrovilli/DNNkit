from Functions import ReadArgParser, checkCreateDir, ReadConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

from colorama import init, Fore
init(autoreset = True)

def find(str_jets,df):
    n=0
    for i in df['origin']==str_jets:
        if i==True:
            n+=1
    return n
    
def composition_plot(df, directory, signal, jetCollection, analysis, channel, PreselectionCuts, background):
    samples = []
    x=np.array([])
    samples=list(set(df['origin']))
    for var in samples:
        x=np.append(x,find(var,df))
    #plt.figure(1,figsize=(18,6))
    plt.bar(samples,x)
    legend = 'jet collection: ' + jetCollection + '\nanalysis: ' + analysis + '\nchannel: ' + channel + '\npreselection cuts: ' + PreselectionCuts
    plt.figtext(0.6, 0.7, legend, wrap = True, horizontalalignment = 'left')#, fontsize = 10)
    plt.suptitle('Dataset composition')
    plt.xlabel('Origin')
    plt.ylabel('Number of events')
    plt.yscale('log')
    pltName = directory + '/' + jetCollection + '_' + analysis + '_' + channel + '_' + signal + '_' + preselectionCuts + '_' + background + '_composition.pdf'
    plt.savefig(pltName)
    print('Saved ' + pltName)
    plt.clf()
    return x,samples

### Reading from command line
jetCollection, analysis, channel, preselectionCuts, background, trainingFraction = ReadArgParser()
print(background)

### Reading from configuration file
dfPath, InputFeatures, signalsList, backgroundsList = ReadConfig(analysis, jetCollection)
dfPath += analysis + '/' + channel

### Adding useful variables to the list of input variables
extendedInputFeatures = InputFeatures.copy()
extendedInputFeatures.append('isSignal')
extendedInputFeatures.append('origin')

### Loading input file
data = pd.read_pickle(dfPath + '/MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '.pkl') 
data = data[extendedInputFeatures]

for signal in signalsList:

    ### Creating output directory
    outputDir = dfPath + '/' + signal
    print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

    ### Creating the list of backgrounds and signal processes to select
    if background == 'all':
        inputOrigin = backgroundsList.copy()
    else:
        inputOrigin = list(background.split('_'))
    inputOrigin.append(signal)
    print(inputOrigin)

    ### Selecting events according to their origin 
    data_set = data[data['origin'].isin(inputOrigin)]

    ### Plotting the dataframe composition
    composition = composition_plot(data_set, outputDir, signal, jetCollection, analysis, channel, preselectionCuts, background)
    
    ### Saving the dataframe
    outputFileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + signal + '_' + preselectionCuts + '_' + background + '_' + str(trainingFraction) + 't'
    dataSetName = outputDir + '/inputDataSet_' + outputFileCommonName + '.pkl'
    data_set.to_pickle(dataSetName)
    print('Saved ' + dataSetName)

    ### Creating x and y arrays 
    X_input = data_set[InputFeatures].values
    y_input = np.array(data_set['isSignal'])
    massColumnIndex = InputFeatures.index('mass')

    ### Creating train/test arrays
    X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, train_size = trainingFraction)

    ### Saving unscaled mass values
    m_train_unscaled = X_train[:, massColumnIndex]
    m_test_unscaled = X_test[:, massColumnIndex]

    ### Scaling train/test arrays
    scaler_train = StandardScaler().fit(X_train)
    X_train = np.array(scaler_train.transform(X_train), dtype=object)
    X_test = np.array(scaler_train.transform(X_test), dtype=object)
          
    ### Saving dataframes as csv files
    np.savetxt(outputDir + '/X_train_' + outputFileCommonName + '.csv', X_train, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/X_train_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/X_test_' + outputFileCommonName + '.csv', X_test, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/X_test_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/y_train_' + outputFileCommonName + '.csv', y_train, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/y_train_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/y_test_' + outputFileCommonName + '.csv', y_test, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/y_test_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/m_train_unscaled_' + outputFileCommonName + '.csv', m_train_unscaled, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/m_train_unscaled_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/m_test_unscaled_' + outputFileCommonName + '.csv', m_test_unscaled, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/m_test_unscaled_' + outputFileCommonName + '.csv')
