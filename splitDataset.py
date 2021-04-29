from argparse import ArgumentParser
import configparser
from Functions import checkCreateDir
from colorama import init, Fore
init(autoreset = True)

parser = ArgumentParser()
parser.add_argument('-t', '--Training', help = 'Relative size of the training sample, between 0 and 1', default = 0.7)
parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'')

args = parser.parse_args()

analysis = args.Analysis
if args.Analysis is None:
    parser.error('Requested type of analysis (either \'mergered\' or \'resolved\')')
elif args.Analysis != 'resolved' and args.Analysis != 'merged':
    parser.error('Analysis can be either \'merged\' or \'resolved\'')
channel = args.Channel
if args.Channel is None:
    parser.error('Requested channel (either \'ggF\' or \'VBF\')')
elif args.Channel != 'ggF' and args.Channel != 'VBF':
    parser.error('Channel can be either \'ggF\' or \'VBF\'')
trainingFraction = float(args.Training)
if args.Training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error('Training fraction must be between 0 and 1')

### Reading from config file
config = configparser.ConfigParser()
config.read('Configuration.txt')
dfPath = config.get('config', 'dfPath')
print (format('Output directory: ' + Fore.GREEN + dfPath), checkCreateDir(dfPath))

### Creating logFile
logFileName = dfPath + 'splitDataSetLogFile_' + analysis + '_' + channel + '.txt'
logFile = open(logFileName, 'w')
logFile.write('Analysis: ' + analysis + '\nChannel: ' + channel)

#### Loading data
import pandas as pd

df_total = pd.read_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '.pkl')

### Splitting sample
Ntrain_stop = int(round(df_total.shape[0] * trainingFraction))
X_Train = df_total[:Ntrain_stop]
X_Test = df_total[Ntrain_stop:]
X_Train.to_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_' + 'Train.pkl')
print('Saved ' + dfPath + 'MixData_PD_' + analysis + '_' + channel + '_' + 'Train.pkl')
print(X_Train[:10])
X_Test.to_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_' + 'Test.pkl')
print('Saved ' + dfPath + 'MixData_PD_' + analysis + '_' + channel + '_' + 'Test.pkl')
print(X_Test[:10])

print('Size of the training saple: ', X_Train.shape)
print('Size of the testing saple: ', X_Test.shape)

logFile.write('\nSize of the training saple: ' + str(X_Train.shape[0]))
logFile.write('\nSize of the testing saple: '+ str(X_Test.shape[0]))
print('Saved ' + logFileName)
logFile.close()
