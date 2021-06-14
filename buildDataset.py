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
    missing_var=np.array([])
    if str(i+'_DF.pkl') not in os.listdir(dfPath):
        print(str(i+'_DF.pkl'),'not in ', dfPath)
        counter+=1
        continue
    print(i)
    inFile = dfPath + i + '_DF.pkl'

    print('Loading ' + inFile)
    logFile.write(i + '_DF.pkl')
    if counter != (len(inputFiles) - 1):
        logFile.write(', ')
    else:
        logFile.write(']')
    newDf = pd.read_pickle(inFile)

    for var in rootBranchSubSample:
        if var not in newDf.columns:
            missing_var=np.append(missing_var,var)
#                print("NO",var)
    if np.size(missing_var)!=0:
        print("Found missing variables in ",i)
        print(missing_var)
        continue

    newDf = newDf[rootBranchSubSample]
    newDf=newDf.assign(origin=re.search('(.+?)-',i).group(1))

    if PreselectionCuts != '':
        newDf = newDf.query(PreselectionCuts)
    if channel == 'ggF':
        newDf = newDf.query('Pass_isVBF == False')
        if analysis == 'merged':
            selection = 'Pass_MergHP_GGF_ZZ_Tag_SR == True or Pass_MergHP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_WZ_SR == True or Pass_MergLP_GGF_ZZ_Tag_SR == True or Pass_MergLP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_ZZ_Tag_ZCR == True or Pass_MergHP_GGF_WZ_ZCR == True or Pass_MergHP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_ZZ_Tag_ZCR == True or Pass_MergLP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_WZ_SR == True or Pass_MergLP_GGF_WZ_ZCR == True'
        elif analysis == 'resolved':
            selection = 'Pass_Res_GGF_WZ_SR == True or Pass_Res_GGF_WZ_ZCR == True or Pass_Res_GGF_ZZ_Tag_SR == True or Pass_Res_GGF_ZZ_Untag_SR == True or Pass_Res_GGF_ZZ_Tag_ZCR == True or Pass_Res_GGF_ZZ_Untag_ZCR == True'
        newDf = newDf.query(selection)
    elif channel == 'VBF':
        newDf = newDf.query('Pass_isVBF == True')        
        if analysis == 'merged':
            selection = 'Pass_MergHP_VBF_WZ_SR == True or Pass_MergHP_VBF_ZZ_SR == True or Pass_MergHP_VBF_WZ_ZCR == True or Pass_MergHP_VBF_ZZ_ZCR == True or Pass_MergLP_VBF_WZ_SR == True or Pass_MergLP_VBF_ZZ_SR == True or Pass_MergLP_VBF_WZ_ZCR == True or Pass_MergLP_VBF_ZZ_ZCR == True' 
        elif analysis == 'resolved':
            selection = 'Pass_Res_VBF_WZ_SR == True or Pass_Res_VBF_WZ_ZCR == True or Pass_Res_VBF_ZZ_SR == True or Pass_Res_VBF_ZZ_ZCR'
        newDf = newDf.query(selection)

    print('done analysis')
    if (dataType[counter] == 'bkg'):
        newDf.insert(len(newDf.columns), 'isSignal', np.zeros(newDf.shape[0]), True)
        newDf.insert(len(newDf.columns), 'mass', np.random.choice(mass,newDf.shape[0]), True)

    if (dataType[counter] == 'sig'):
        newDf.insert(len(newDf.columns), 'isSignal', np.ones(newDf.shape[0]), True)

        masses=np.zeros(len(newDf['DSID']))
        DSID_values=set(newDf['DSID'])
        print('DSID values:',DSID_values)
        for x in DSID_values:
            found = False
            print('searching for ', x,' DSID')
            for j in range(len(DSID)):
                if (x == int(DSID[j])):
                    found=True
                    print('found mass:',int(mass[j]))
                    masses[np.where(newDf['DSID']==x)]=mass[j]
                    continue
            if found==False:
                print('mass related to',x,'not found')
                masses[np.where(newDf['DSID']==x)]=0

        print('found masses:',set(masses))
        newDf.insert(len(newDf.columns), 'mass', masses, True)

    print(newDf[0:5])
    df.append(newDf)
    counter+=1

df_pd = pd.DataFrame()
for i in range(len(df)):
    df_pd = pd.concat([df_pd, df[i]], ignore_index = True)

### Shuffling data
df_pd = ShufflingData(df_pd)

logFile.write('\nNumber of events: ' + str(df_pd.shape[0]))

print('Number of events:',str(df_pd.shape[0]))
print('Saved ' + logFileName)
logFile.close()

### Saving pkl files
#df_pd.to_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '.pkl')
import pickle
with open(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_p4.pkl', 'wb') as output_file:
    pickle.dump(df_pd, output_file, protocol=4)
print('Saved to ' + dfPath + 'MixData_PD_' + analysis + '_' + channel + '_p4.pkl')

#cut=[var+str('!=-99 and') for var in dataVariables ]
#flatten_cut=' '.join(cut)
#flatten_cut=flatten_cut[:(len(flatten_cut)-4)]

#df_pd_cut=df_pd.query(flatten_cut)

#df_pd_cut.to_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + 'cut.pkl')
