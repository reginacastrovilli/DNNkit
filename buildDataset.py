import pandas as pd
import numpy as np
import argparse, configparser
import re
import ast
import random
from Functions import checkCreateDir, ShufflingData
import os.path
from colorama import init, Fore
init(autoreset = True)

parser = argparse.ArgumentParser(description = 'Deep Neural Network Training and testing Framework')
parser.add_argument('-p', '--Preselection', default = '', help = 'String which will be translated to python command to filter the initial PDs according to it. E.g. \'lep1_pt > 0 and lep1_eta > 0\'', type = str)
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
PreselectionCuts = args.Preselection

### Reading from config file
config = configparser.ConfigParser()
config.read('Configuration.txt')
inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
dataType = ast.literal_eval(config.get('config', 'dataType'))
rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
dfPath = config.get('config', 'dfPath')
print (format('Output directory: ' + Fore.GREEN + dfPath), checkCreateDir(dfPath))

if len(inputFiles) != len(dataType):
    print(format(Fore.RED + 'Data type array does not match input files array'))
    exit()

### Loading DSID-mass map
f = open('DSIDtoMass.txt')
lines = f.readlines()
DSID = [int(i.split(':')[0]) for i in lines]
mass = [int(i.split(':')[1]) for i in lines]

### Loading pkl files, selecting only relevant variables, creating sig/bkg flag, converting DSID into mass
df = []
counter = 0
logFileName = dfPath + 'buildDataSetLogFile_' + analysis + '_' + channel + '.txt'
logFile = open(logFileName, 'w')
logFile.write('Analysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + PreselectionCuts + '\nInput files path: ' + dfPath + '\nrootBranchSubSamples: ' + str(rootBranchSubSample) + '\nInput files: [')

for i in inputFiles:
    inFile = dfPath + i + '_DF.pkl'
    if os.path.exists(inFile) == False:
        print(format(Fore.RED + 'File ' + inFile + ' does not exist'))
        counter+=1
        continue
    print('Loading ' + inFile)
    logFile.write(i + '_DF.pkl')
    if counter != (len(inputFiles) - 1):
        logFile.write(', ')
    else:
        logFile.write(']')
    newDf = pd.read_pickle(inFile)
    newDf = newDf[rootBranchSubSample]
                
    if PreselectionCuts != '':
        newDf = newDf.query(PreselectionCuts)
    if analysis == 'merged':
        selection = 'Pass_MergHP_GGF_ZZ_Tag_SR == True or Pass_MergHP_GGF_ZZ_Untag_SR == True or Pass_MergLP_GGF_ZZ_Tag_SR == True or Pass_MergLP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_ZZ_Tag_ZCR == True or Pass_MergHP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_ZZ_Tag_ZCR == True or Pass_MergLP_GGF_ZZ_Untag_ZCR == True'
        newDf = newDf.query(selection)
    elif analysis == 'resolved':
        selection = 'Pass_MergHP_GGF_ZZ_Tag_SR == False and Pass_MergHP_GGF_ZZ_Untag_SR == False and Pass_MergLP_GGF_ZZ_Tag_SR == False and Pass_MergLP_GGF_ZZ_Untag_SR == False and Pass_MergHP_GGF_ZZ_Tag_ZCR == False and Pass_MergHP_GGF_ZZ_Untag_ZCR == False and Pass_MergLP_GGF_ZZ_Tag_ZCR == False and Pass_MergLP_GGF_ZZ_Untag_ZCR == False'
        newDf = newDf.query(selection)
    if channel == 'ggF':
        newDf = newDf.query('Pass_isVBF == False')
    elif channel == 'VBF':
        newDf = newDf.query('Pass_isVBF == True')
    newDf.insert(len(newDf.columns), "mass", np.zeros(newDf.shape[0]), True)
    if (dataType[counter] == 'sig'):
        newDf.insert(len(newDf.columns), "isSignal", np.ones(newDf.shape[0]), True)
        for k in range(newDf.shape[0]):
            found = False
            for j in range(len(DSID)):
                if (newDf.iat[k, 0] == int(DSID[j])):               #if iat -> at we can address the column by name, so the order of the variables won't
                    newDf.iat[k, newDf.shape[1] - 2] = int(mass[j]) #matter anymore. However, the code will really be slower
                    found = True
            if (found == False):
                print(format(Fore.RED + 'WARNING !!! missing mass value for DSID ' + str(newDf.iat[k,0])))
    else:
        newDf.insert(len(newDf.columns), "isSignal", np.zeros(newDf.shape[0]), True)
        ### Assigning to background events a random signal mass
        for event in range(newDf.shape[0]):
            newDf.iat[event, newDf.shape[1] - 2] = random.choice(mass)
    print(newDf[0:20])
    df.append(newDf)
    counter+=1

df_pd = pd.DataFrame()
for i in range(len(df)):
    df_pd = pd.concat([df_pd, df[i]], ignore_index = True)

### Shuffling data
df_pd = ShufflingData(df_pd)

logFile.write('\nNumber of events: ' + str(df_pd.shape[0]))                                                                                  
print('Saved ' + logFileName)
logFile.close()

### Saving pkl files
df_pd.to_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '.pkl')
print('Saved to ' + dfPath + 'MixData_PD_' + analysis + '_' + channel + '.pkl')
