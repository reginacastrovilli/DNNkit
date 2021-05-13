from argparse import ArgumentParser
import configparser
from Functions import checkCreateDir, ShufflingData
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from colorama import init, Fore
init(autoreset = True)

parser = ArgumentParser()
parser.add_argument('-t', '--Training', help = 'Relative size of the training sample, between 0 and 1', default = 0.7)
parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'')

args = parser.parse_args()

analysis = args.Analysis
if args.Analysis is None:
    parser.error(Fore.RED + 'Requested type of analysis (either \'mergered\' or \'resolved\')')
elif args.Analysis != 'resolved' and args.Analysis != 'merged':
    parser.error(Fore.RED + 'Analysis can be either \'merged\' or \'resolved\'')
channel = args.Channel
if args.Channel is None:
    parser.error(Fore.RED + 'Requested channel (either \'ggF\' or \'VBF\')')
elif args.Channel != 'ggF' and args.Channel != 'VBF':
    parser.error(Fore.RED + 'Channel can be either \'ggF\' or \'VBF\'')
trainingFraction = float(args.Training)
if args.Training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error(Fore.RED + 'Training fraction must be between 0 and 1')

### Reading the configuration file
config = configparser.ConfigParser()
config.read('Configuration.txt')
dfPath = config.get('config', 'dfPath')
#print ('Output directory: ' + Fore.GREEN + dfPath, checkCreateDir(dfPath))
'''
### Creating logFile
logFileName = dfPath + 'splitDataSetLogFile_' + analysis + '_' + channel + '.txt'
logFile = open(logFileName, 'w')
logFile.write('Analysis: ' + analysis + '\nChannel: ' + channel)
'''
#### Loading data
import pandas as pd

df_total = pd.read_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '.pkl')

### Splitting signal from background
y = df_total['isSignal']

df_signal = df_total[y == 1]
df_bkg = df_total[y != 1]

### Saving signal mass values
signal_mass = df_signal['DSID']
massPoints = list(set(signal_mass))

### Looping over signal mass value
import numpy as np

for mass in massPoints:

    ### Creating a dataframe with single mass signal events
    df_signal_mass = df_signal[signal_mass == mass]
    print(df_signal_mass.shape[0])
    ### Merging the previous dataframe with background events
    df_mass = pd.concat([df_signal_mass, df_bkg], ignore_index = True)

    ### Shuffling data
    df_mass = ShufflingData(df_mass)

    ### Scaling data except IsSignal
    transformer = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, df_mass.shape[1] - 1))], remainder = 'passthrough')
    df_mass = pd.DataFrame(transformer.fit_transform(df_mass), index = df_mass.index, columns = df_mass.columns)

    ### Splitting into train/test sample
    Ntrain_stop = int(round(df_mass.shape[0] * trainingFraction))
    X_Train_mass = df_mass[:Ntrain_stop]
    X_Test_mass = df_mass[Ntrain_stop:]

    ### Creating the output directory
    outputDir = 'NN_SplitData/' + str(mass)
    print ('Output directory: ' + Fore.GREEN + outputDir, checkCreateDir(outputDir))

    ### Saving train and test dataframes
    X_Train_mass.to_pickle(outputDir + '/MixData_PD_' + analysis + '_' + channel + '_' + 'Train.pkl')
    print('Saved ' + outputDir + '/MixData_PD_' + analysis + '_' + channel + '_' + 'Train.pkl')
    X_Test_mass.to_pickle(outputDir + '/MixData_PD_' + analysis + '_' + channel + '_' + 'Test.pkl')
    print('Saved ' + outputDir + '/MixData_PD_' + analysis + '_' + channel + '_' + 'Test.pkl')
