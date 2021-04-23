from argparse import ArgumentParser
import configparser

parser = ArgumentParser()
parser.add_argument('-t', '--training', help = 'Relative size of the training sample, between 0 and 1', default = 0.7)
parser.add_argument('-a', '--Analysis', default = 'all', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', default = 'all', help = 'Channel: \'ggF\' or \'VBF\'')

args = parser.parse_args()
analysis = args.Analysis
channel = args.Channel
trainingFraction = float(args.training)

if args.training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error('Training fraction must be between 0 and 1')
if args.Analysis and (analysis != 'merged' or 'resolved'):
    parser.error('Analysis can be either \'merged\' or \'resolved\'')
if args.Channel and (channel != 'ggF' or 'VBF'):
    parser.error('Channel can be either \'ggF\' or \'VBF\'')
#logFile = open('logFile.txt', 'a')
#logFile.write('\nTraining fraction: ' + str(trainingFraction))
#logFile.close()

### Reading from config file
config = configparser.ConfigParser()
config.read('Config.txt')
dfPath = config.get('config', 'dfPath')

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
