from Functions import ReadArgParser, ReadConfig, checkCreateDir, ShufflingData
import pandas as pd
import numpy as np
import re
import ast
import random
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
DSIDfile = open('DSIDtoMass.txt')
lines = DSIDfile.readlines()
DSID = [int(line.split(':')[0]) for line in lines]
mass = [int(line.split(':')[1]) for line in lines]

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

for inputFile in inputFiles:
    missing_var = np.array([])
    if str(inputFile + '_DF.pkl') not in os.listdir(dfPath):
        print(str(inputFile + '_DF.pkl'), 'not in ', dfPath)
        counter += 1
        continue
    ### Loading input file
    inFile = dfPath + inputFile + '_DF.pkl'
    print(Fore.GREEN + 'Loading ' + inFile)
    logFile.write(inputFile + '_DF.pkl')
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

    ### Creating a new dataframe with only the desired variables
    newDf = newDf[rootBranchSubSample]

    ### Adding a new column in the dataframe with the name of the process for each event
    if (re.search('(.+?)-', inputFile) == None):
        originName = inputFile
    else:
        originName = re.search('(.+?)-', inputFile).group(1)
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
        print('signal')
       
        ### Selecting signal events with mass in the mass sublist
        newDf = newDf[newDf['DSID'].isin(DSIDList)]
        ### Inserting new 'isSignal' column with value 1
        newDf.insert(len(newDf.columns), 'isSignal', np.ones(newDf.shape[0]), True)
        ### Creating an array of mass values corresponding to the values in the 'DSID' column
        masses = np.zeros(len(newDf['DSID']))
        DSIDvalues = set(newDf['DSID'])
        print(Fore.BLUE + 'DSID values in the dataset: ' + str(DSIDvalues))
        for DSIDval in DSIDvalues:
            found = False
            print(Fore.BLUE + 'Searching for mass associated to DSID = ' + str(DSIDval) + '...')
            for j in range(len(DSID)):
                if (DSIDval == int(DSID[j])):
                    found=True
                    print(Fore.BLUE + '... found mass = ' +  str(int(mass[j])))
                    masses[np.where(newDf['DSID'] == DSIDval)] = mass[j]
                    continue
            if found == False:
                print(Fore.RED + '... mass not found!')
        print(Fore.BLUE + 'Masses found in the dataset: ' + str(set(masses)))
        ### Inserting the new array as a new 'mass' column in the dataframe
        newDf.insert(len(newDf.columns), 'mass', masses, True)

    ### Selecting in the dataframe only the variables that will be given as input to the neural networks
    newDf = newDf[InputFeatures]

    df.append(newDf)
    counter+=1

pandasDF = pd.DataFrame()
for i in range(len(df)):
    pandasDF = pd.concat([pandasDF, df[i]], ignore_index = True)

### Shuffling data
pandasDF = ShufflingData(pandasDF)

logFile.write('\nNumber of events: ' + str(pandasDF.shape[0]))
logFile.close()
print('Saved ' + logFileName)

### Saving pkl files
outputDir = dfPath + analysis + '/' + channel
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
outputFileName = '/MixData_PD_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '.pkl'
pandasDF.to_pickle(outputDir + outputFileName)
print('Saved ' + outputDir + outputFileName)
