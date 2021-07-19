from Functions import ReadArgParser, checkCreateDir, ReadConfig, SaveFeatureScaling
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
#signalsList = ['Radion']
### Adding useful variables to the list of input variables
extendedInputFeatures = InputFeatures.copy()
extendedInputFeatures.append('isSignal')
extendedInputFeatures.append('origin')

### Loading input file
data = pd.read_pickle(dfPath + '/MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '.pkl') 
data = data[extendedInputFeatures]

for signal in signalsList:

    ### Creating output directory
    outputDir = dfPath + '/' + signal + '/' + background
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
    origin_input = np.array(data_set['origin'])
    massColumnIndex = InputFeatures.index('mass')

    ### Creating train/test arrays
    X_train, X_test, y_train, y_test, origin_train, origin_test = train_test_split(X_input, y_input, origin_input, train_size = trainingFraction)
    #origin_train = pd.DataFrame(origin_train)
    #origin_test = pd.DataFrame(origin_test)

    origin_train = np.array(origin_train)
    print(origin_train)
    origin_train = np.where(origin_train == 'Radion', 0, origin_train)
    origin_train = np.where(origin_train == 'Zjets', 1, origin_train)
    origin_train = np.where(origin_train == 'Diboson', 2, origin_train)
    print(origin_train)

    origin_test = np.array(origin_test)
    origin_test = np.where(origin_test == 'Radion', 0, origin_test)
    origin_test = np.where(origin_test == 'Zjets', 1, origin_test)
    origin_test = np.where(origin_test == 'Diboson', 2, origin_test)


    np.savetxt('/nfs/kloe/einstein4/HDBS/PDNNTest_InputDataFrames/TCC/merged/ggF/Signal/X_train_unscaled.txt', X_train, delimiter = ',', fmt = '%s')
    np.savetxt('/nfs/kloe/einstein4/HDBS/PDNNTest_InputDataFrames/TCC/merged/ggF/Signal/X_test_unscaled.txt', X_test, delimiter = ',', fmt = '%s')

    ### Saving unscaled mass values and train sample
    m_train_unscaled = X_train[:, massColumnIndex]
    m_test_unscaled = X_test[:, massColumnIndex]
    X_train_unscaled = X_train.copy()

    ### Scaling train/test arrays
    scaler_train = StandardScaler().fit(X_train)
    #print(scaler_train.mean_)
    #print(scaler_train.var_)
    X_train = np.array(scaler_train.transform(X_train), dtype=object)
    X_test = np.array(scaler_train.transform(X_test), dtype=object)

    np.savetxt('/nfs/kloe/einstein4/HDBS/PDNNTest_InputDataFrames/TCC/merged/ggF/Signal/X_train_scaled.txt', X_train, delimiter = ',', fmt = '%s')
    np.savetxt('/nfs/kloe/einstein4/HDBS/PDNNTest_InputDataFrames/TCC/merged/ggF/Signal/X_test_scaled.txt', X_test, delimiter = ',', fmt = '%s')

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
    np.savetxt(outputDir + '/origin_train_' + outputFileCommonName + '.csv', origin_train, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/origin_train_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/origin_test_' + outputFileCommonName + '.csv', origin_test, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/origin_test_' + outputFileCommonName + '.csv')
    np.savetxt(outputDir + '/X_train_unscaled_' + outputFileCommonName + '.csv', X_train_unscaled, delimiter = ',', fmt = '%s')
    print('Saved ' + outputDir + '/X_train_unscaled_' + outputFileCommonName + '.csv')
