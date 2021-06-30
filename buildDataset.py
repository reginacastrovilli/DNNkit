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
parser.add_argument('-p', '--PreselectionCuts', help = 'String which will be translated to python command to filter the initial PDs according to it. E.g. \'lep1_pt > 0 and lep1_eta > 0\'', type = str, default = 'none') ### cambiare?
parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'')
parser.add_argument('-j', '--JetCollection', help = 'Jet collection: \'TCC\'', type = str, default = 'TCC')
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
PreselectionCuts = args.PreselectionCuts
if args.PreselectionCuts == 'none': 
    print(Fore.BLUE + 'Preselection cuts not specified, applying: ' + str(PreselectionCuts))
jetCollection = args.JetCollection
if args.JetCollection is None:
    parser.error(Fore.RED + 'Requested jet collection (\'TCC\' or )')
elif args.JetCollection != 'TCC':
    parser.error(Fore.RED + 'Jet collection can be \'TCC\', ')

### Reading from config file
config = configparser.ConfigParser()
config.read('newConfiguration.ini')
inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
dataType = ast.literal_eval(config.get('config', 'dataType'))
rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
if analysis == 'merged':
    InputFeatures = ast.literal_eval(config.get('config', 'InputFeaturesMerged'))
elif analysis == 'resolved':
    InputFeatures = ast.literal_eval(config.get('config', 'InputFeaturesResolved'))
dfPath = config.get('config', 'dfPath')
dfPath += jetCollection + '_DataFrames/' ### toglierei _DataFrames
print (format('Output directory: ' + Fore.GREEN + dfPath), checkCreateDir(dfPath)) ### non ci deve essere per forza? Perché è la directory da cui carico i file

if len(inputFiles) != len(dataType):
    print(format(Fore.RED + 'Data type array does not match input files array'))
    exit()

### Loading DSID-mass map
f = open('DSIDtoMass.txt')
lines = f.readlines()
DSID = [int(i.split(':')[0]) for i in lines]
mass = [int(i.split(':')[1]) for i in lines]

### Creating new DSID/mass lists for the merged and resolved analysis
DSIDList = []
massList = []
counter = 0

for massValue in mass:
    if analysis == 'merged' and massValue >= 500:
        massList.append(massValue)
        DSIDList.append(DSID[counter])
    if analysis == 'resolved' and massValue <= 1500:
        massList.append(massValue)
        DSIDList.append(DSID[counter])
    counter += 1

### Defining selections for different analysis and channel
selectionMergedGGF = 'Pass_MergHP_GGF_ZZ_Tag_SR == True or Pass_MergHP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_WZ_SR == True or Pass_MergLP_GGF_ZZ_Tag_SR == True or Pass_MergLP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_ZZ_Tag_ZCR == True or Pass_MergHP_GGF_WZ_ZCR == True or Pass_MergHP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_ZZ_Tag_ZCR == True or Pass_MergLP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_WZ_SR == True or Pass_MergLP_GGF_WZ_ZCR == True'
selectionMergedVBF = 'Pass_MergHP_VBF_WZ_SR == True or Pass_MergHP_VBF_ZZ_SR == True or Pass_MergHP_VBF_WZ_ZCR == True or Pass_MergHP_VBF_ZZ_ZCR == True or Pass_MergLP_VBF_WZ_SR == True or Pass_MergLP_VBF_ZZ_SR == True or Pass_MergLP_VBF_WZ_ZCR == True or Pass_MergLP_VBF_ZZ_ZCR == True' 
selectionResolvedGGF = 'Pass_Res_GGF_WZ_SR == True or Pass_Res_GGF_WZ_ZCR == True or Pass_Res_GGF_ZZ_Tag_SR == True or Pass_Res_GGF_ZZ_Untag_SR == True or Pass_Res_GGF_ZZ_Tag_ZCR == True or Pass_Res_GGF_ZZ_Untag_ZCR == True'
selectionResolvedVBF = 'Pass_Res_VBF_WZ_SR == True or Pass_Res_VBF_WZ_ZCR == True or Pass_Res_VBF_ZZ_SR == True or Pass_Res_VBF_ZZ_ZCR'

### Loading pkl files, selecting only relevant variables, creating sig/bkg flag, converting DSID into mass
df = []
counter = 0
logFileName = dfPath + 'buildDataSetLogFile_' + analysis + '_' + channel + '.txt' ### cambiare
logFile = open(logFileName, 'w')
logFile.write('Analysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + PreselectionCuts + '\nInput files path: ' + dfPath + '\nrootBranchSubSamples: ' + str(rootBranchSubSample) + '\nInput files: [')

for i in inputFiles:
    missing_var=np.array([])
    if str(i+'_DF.pkl') not in os.listdir(dfPath):
        print(str(i+'_DF.pkl'),'not in ', dfPath)
        counter+=1
        continue
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
    newDf=newDf.assign(origin=re.search('(.+?)-',i).group(1)) ###### ?

    if PreselectionCuts != 'none':
        newDf = newDf.query(PreselectionCuts)
    if channel == 'ggF':
        newDf = newDf.query('Pass_isVBF == False')
        if analysis == 'merged':
            selection = selectionMergedGGF
        elif analysis == 'resolved':
            selection = selectionResolvedGGF
    elif channel == 'VBF':
        newDf = newDf.query('Pass_isVBF == True')        
        if analysis == 'merged':
            selection = selectionMergedVBF
        elif analysis == 'resolved':
            selection = selectionResolvedVBF
    newDf = newDf.query(selection)

    if (dataType[counter] == 'bkg'):
        ### Inserting new 'isSignal' column with value 0
        newDf.insert(len(newDf.columns), 'isSignal', np.zeros(newDf.shape[0]), True)
        ### Insertign new 'mass' column with value randomly selected from the mass sublist
        newDf.insert(len(newDf.columns), 'mass', np.random.choice(massList, newDf.shape[0]), True) ###

    if (dataType[counter] == 'sig'):
        ### Selecting signal events with mass in the mass sublist
        newDf = newDf[newDf['DSID'].isin(DSIDList)]
        ### Inserting new 'isSignal' column with value 1
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

        print('found masses:',set(masses))
        newDf.insert(len(newDf.columns), 'mass', masses, True)

    newDf = newDf[InputFeatures]
    print(newDf[0:5])
    df.append(newDf)
    counter+=1

df_pd = pd.DataFrame()
for i in range(len(df)):
    df_pd = pd.concat([df_pd, df[i]], ignore_index = True)
print(df_pd.shape[0])

### Shuffling data
df_pd = ShufflingData(df_pd)

logFile.write('\nNumber of events: ' + str(df_pd.shape[0]))
logFile.close()
print('Saved ' + logFileName)

### Saving pkl files
outputDir = dfPath + jetCollection + '/' + analysis + '/' + channel
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
outputFileName = 'MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + PreselectionCuts + '.pkl'
df_pd.to_pickle(outputDir + outputFileName)
print('Saved ' + outputDir + outputFileName)
