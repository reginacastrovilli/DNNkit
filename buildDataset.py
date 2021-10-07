import pandas as pd
import numpy as np
import argparse, configparser
import re
import ast
import random
from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData
import os.path
from colorama import init, Fore
init(autoreset = True)

### Reading the command line
jetCollection, analysis, channel, preselectionCuts = ReadArgParser()

### Reading from config file
inputFiles, dataType, rootBranchSubSample, dfPath, InputFeatures = ReadConfig(analysis, jetCollection)
InputFeatures.append('isSignal')
InputFeatures.append('origin')

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
logFile.write('Analysis: ' + analysis + '\nChannel: ' + channel + '\nPreselection cuts: ' + preselectionCuts + '\nInput files path: ' + dfPath + '\nrootBranchSubSamples: ' + str(rootBranchSubSample) + '\nInput files: [')

for i in inputFiles:
    missing_var=np.array([])
    if str(i+'_DF.pkl') not in os.listdir(dfPath):
        print(str(i+'_DF.pkl'),'not in ', dfPath)
        counter+=1
        continue
    ### Loading input file
    inFile = dfPath + i + '_DF.pkl'
    print(Fore.GREEN + 'Loading ' + inFile)
    logFile.write(i + '_DF.pkl')
    if counter != (len(inputFiles) - 1):
        logFile.write(', ')
    else:
        logFile.write(']')
    newDf = pd.read_pickle(inFile)

    ### Checking that all input variables are in the input file
    for var in rootBranchSubSample:
        if var not in newDf.columns:
            missing_var=np.append(missing_var,var)
    if np.size(missing_var)!=0:
        print("Found missing variables in ",i)
        print(missing_var)
        continue

    ### Creating a new dataframe with only the input variables
    newDf = newDf[rootBranchSubSample]

    ### Adding a new column in the dataframe with the name of the process for each event
    if (re.search('(.+?)-', i) == None):
        originName = i
    else:
        originName = re.search('(.+?)-', i).group(1)
    newDf = newDf.assign(origin = originName)

    ### Applying preselection cuts
    if preselectionCuts != 'none':
        newDf = newDf.query(preselectionCuts)
        
    ### Selecting events for each type of analysis and channel
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
        ### Inserting new 'mass' column with value randomly selected from the mass sublist
        newDf.insert(len(newDf.columns), 'mass', np.random.choice(massList, newDf.shape[0]), True) ###

    if (dataType[counter] == 'sig'):
        ### Selecting signal events with mass in the mass sublist
        newDf = newDf[newDf['DSID'].isin(DSIDList)]
        ### Inserting new 'isSignal' column with value 1
        newDf.insert(len(newDf.columns), 'isSignal', np.ones(newDf.shape[0]), True)
        ### Creating an array of mass values corresponding to the values in the 'DSID' column
        masses = np.zeros(len(newDf['DSID']))
        DSID_values = set(newDf['DSID'])
        print(Fore.BLUE + 'DSID values in the dataset: ' + str(DSID_values))
        for x in DSID_values:
            found = False
            print(Fore.BLUE + 'Searching for mass associated to DSID = ' + str(x) + '...')
            for j in range(len(DSID)):
                if (x == int(DSID[j])):
                    found=True
                    print(Fore.BLUE + '... found mass = ' +  str(int(mass[j])))
                    masses[np.where(newDf['DSID']==x)]=mass[j]
                    continue
            if found==False:
                print(Fore.RED + '... mass not found!')
        print(Fore.BLUE + 'Masses found in the dataset: ' + str(set(masses)))
        ### Inserting the new array as a new 'mass' column in the dataframe
        newDf.insert(len(newDf.columns), 'mass', masses, True)

    ### Selecting in the dataframe only the variables that will be given as input to the neural networks
    newDf = newDf[InputFeatures]

    #print(newDf[0:5])
    df.append(newDf)
    counter+=1
exit()
df_pd = pd.DataFrame()
for i in range(len(df)):
    df_pd = pd.concat([df_pd, df[i]], ignore_index = True)

### Shuffling data
df_pd = ShufflingData(df_pd)

logFile.write('\nNumber of events: ' + str(df_pd.shape[0]))
logFile.close()
print('Saved ' + logFileName)

### Saving pkl files
outputDir = dfPath + analysis + '/' + channel
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
outputFileName = '/MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '.pkl'
df_pd.to_pickle(outputDir + outputFileName)
print('Saved ' + outputDir + outputFileName)
