from Functions import ReadArgParser, checkCreateDir, ReadConfig, SaveFeatureScaling, DrawCorrelationMatrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
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
    pltName = directory + '/' + jetCollection + '_' + analysis + '_' + channel + '_' + signal + '_' + preselectionCuts + '_' + background + '_composition.png'
    plt.savefig(pltName)
    print('Saved ' + pltName)
    plt.clf()
    return x,samples

### Reading from command line
jetCollection, analysis, channel, preselectionCuts, background, testSignal, trainingFraction = ReadArgParser()

### Reading from configuration file
dfPath, InputFeatures, signalsList, backgroundsList = ReadConfig(analysis, jetCollection) ### inputFeatures non serve più! 
dfPath += analysis + '/' + channel

### Creating the list of signals to take
if testSignal == 'all':
    testSignal = signalsList.copy()
else:
    testSignal = list(testSignal.split('_'))

### Loading input file
data = pd.read_pickle(dfPath + '/MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '.pkl') 

foundSignal = 0
drawPlot = False

for signal in signalsList:
    ### Selecting only the request signal
    if signal not in testSignal:
        continue
    foundSignal += 1

    ### Creating output directory
    outputDir = dfPath + '/' + signal + '/' + background
    print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

    ### Creating the list of backgrounds and signal processes to select
    if background == 'all':
        inputOrigin = backgroundsList.copy()
    else:
        inputOrigin = list(background.split('_'))
    inputOrigin.append(signal)

    ### Selecting events according to their origin 
    data_set = data[data['origin'].isin(inputOrigin)]

    if(drawPlot):
        ### Plotting the dataframe composition and the correlation matrix
        composition = composition_plot(data_set, outputDir, signal, jetCollection, analysis, channel, preselectionCuts, background)
        DrawCorrelationMatrix(data_set, InputFeatures, outputDir, jetCollection, analysis, channel, signal, preselectionCuts, background)

    ### Splitting data into train and test set
    data_train, data_test = train_test_split(data_set, train_size = trainingFraction)

    ### Saving unscaled data
    outputFileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + signal + '_' + preselectionCuts + '_' + background + '_' + str(trainingFraction) + 't'
    dataTrainUnscaledName = outputDir + '/data_train_unscaled_' + outputFileCommonName + '.pkl' 
    data_train.to_pickle(dataTrainUnscaledName)
    print('Saved ' + dataTrainUnscaledName)
    m_test_unscaled = data_test['mass']
    mTestUnscaledName = outputDir + '/m_test_unscaled_' + outputFileCommonName + '.pkl' 
    m_test_unscaled.to_pickle(mTestUnscaledName)
    print('Saved ' + mTestUnscaledName)

    ### Saving list of columns names
    columnsNames = data_train.columns

    ### Scaling InputFeatures of train and test set
    ct = ColumnTransformer([('scaler', StandardScaler(), InputFeatures)], remainder='passthrough')
    scaler_train = ct.fit(data_train)
    data_train = scaler_train.transform(data_train)
    data_test = scaler_train.transform(data_test)
    #print(scaler_train.mean_)
    #print(scaler_train.var_)

    ### Converting numpy arrays into pandas dataframes
    data_train = pd.DataFrame(data_train, columns = columnsNames)
    data_test = pd.DataFrame(data_test, columns = columnsNames)

    ### Saving scaled dataframes
    dataTrainName = outputDir + '/data_train_' + outputFileCommonName + '.pkl'
    data_train.to_pickle(dataTrainName)
    print('Saved ' + dataTrainName)
    dataTestName = outputDir + '/data_test_' + outputFileCommonName + '.pkl'
    data_test.to_pickle(dataTestName)
    print('Saved ' + dataTestName)

if foundSignal == 0:
    print(Fore.RED + 'Requested signal not found')
