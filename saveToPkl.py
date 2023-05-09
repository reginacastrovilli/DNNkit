# This script takes the input root files and converts the relevant branches into pkl
# It is not recommended to convert the whole input tree because the script may crash due to memory issues with large trees 

import uproot
#import uproot3
from Functions import ReadArgParser, ReadConfigSaveToPkl
from termcolor import colored, cprint

#### date & time 
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
print("Starting at ", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
print("date and time = ", dt_string)

### Reading the command line
tag = ReadArgParser()

import configparser, ast
import shutil
import os

### Reading from config file
ntuplePath, inputFiles, dfPath, rootBranchSubSample = ReadConfigSaveToPkl(tag)
print('From configuration file: ')
print('inputFiles         : ' + str(inputFiles))
print('rootBranchSubSample: ' + str(rootBranchSubSample))


### Creating log file
logFileName = dfPath + 'EventsNumberNtuples_' + tag + '_AT_'+ dt_string +'.txt'
logFile = open(logFileName, 'w')
print('\nPath to the input nutples: ' + ntuplePath)
print(  'Path to the output df    : ' + dfPath)
logFile.write('saveToPkl.py - Starting at  '+ str(now))
logFile.write('\nCxAOD tag: ' + tag)
logFile.write('\nPath to the input nutples: ' + ntuplePath)
logFile.write('\nNumber of events in the input ntuples:\n')

### Loading, converting and saving each input file
totalEvents = 0 
weightedTotalEvents = 0 

selectionPass = 'Pass_isVBFVV == True or Pass_VV2Lep_Res_GGF_WZ_SR == True or Pass_VV2Lep_Res_VBF_WZ_SR == True or Pass_VV2Lep_Res_VBF_ZZ_SR == True or Pass_VV2Lep_MergHP_GGF_WZ_SR == True or Pass_VV2Lep_MergHP_VBF_WZ_SR == True or Pass_VV2Lep_MergHP_VBF_ZZ_SR == True or Pass_VV2Lep_MergLP_GGF_WZ_SR == True or Pass_VV2Lep_MergLP_VBF_WZ_SR == True or Pass_VV2Lep_MergLP_VBF_ZZ_SR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_SR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_SR == True or Pass_VV2Lep_Res_GGF_WZ_ZCR == True or Pass_VV2Lep_Res_VBF_WZ_ZCR == True or Pass_VV2Lep_Res_VBF_ZZ_ZCR == True or Pass_VV2Lep_MergHP_GGF_WZ_ZCR == True or Pass_VV2Lep_MergHP_VBF_WZ_ZCR == True or Pass_VV2Lep_MergHP_VBF_ZZ_ZCR == True or Pass_VV2Lep_MergLP_GGF_WZ_ZCR == True or Pass_VV2Lep_MergLP_VBF_WZ_ZCR == True or Pass_VV2Lep_MergLP_VBF_ZZ_ZCR == True or Pass_VV2Lep_Res_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_Res_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_MergHP_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_2btag_ZCR == True or Pass_VV2Lep_MergHP_GGF_ZZ_01btag_ZCR == True or Pass_VV2Lep_MergLP_GGF_ZZ_01btag_ZCR'

print('Ready to loop over '+str(len(inputFiles))+' inputFiles ..... ')
#quit()
for i in inputFiles:
    inFile = ntuplePath + i + '.root'
    print('Loading ' + inFile)
    '''
    theFile = uproot3.open(inFile)
    tree = theFile['Nominal']
    Nevents = tree.numentries
    totalEvents += Nevents
    if Nevents == 0:
        print(Fore.RED + 'Ignoring empty file')
        continue
    DF = tree.pandas.df(rootBranchSubSample)
    weightedEvents = DF['weight'].sum()
    weightedTotalEvents += weightedEvents
    print(Fore.BLUE + 'Number of events in ' + inFile, '->\t' + str(Nevents) + ' (weighted: ' + str(weightedEvents) + ')')
    logFile.write(i + ' -> ' + str(Nevents) + ' events (weighted: ' + str(weightedEvents) + ')\n')
    outFile = dfPath + i + '_DF.pkl'
    DF.to_pickle(outFile)
    '''
    with uproot.open(inFile) as theFile:
        tree = theFile['Nominal']
        Nevents = tree.num_entries
        totalEvents += Nevents
        if Nevents == 0:
            cprint('Ignoring empty file', 'red')
            continue

        ### Converting to pandas dataframe only the variables listed in rootBranchSubSample
        DF = tree.arrays(rootBranchSubSample, library = 'pd')

        ### Selecting only events with at least one Pass variable == True
        DF = DF.query(selectionPass)
    
        weightedEvents = DF['weight'].sum()
        weightedTotalEvents += weightedEvents
        cprint('Number of events in ' + inFile + '->\t' + str(Nevents) + ' (weighted: ' + str(weightedEvents) + ')', 'blue')
        logFile.write(i + ' -> ' + str(Nevents) + ' events (weighted: ' + str(weightedEvents) + ')\n')
        outFile = dfPath + i + '_DF.pkl'
        
        ### Saving output dataframe
        DF.to_pickle(outFile)
        cprint('Saved ' + outFile, 'green')

cprint('Total events: ' + str(totalEvents) + ' (weighted: ' + str(weightedTotalEvents) + ')', 'blue')
logFile.write('Number of total events: ' + str(totalEvents) + ' (weighted: ' + str(weightedTotalEvents) + ')\n')
logFile.write('\npkl files saved in ' + dfPath)
logFile.close()
cprint('Saved ' + logFileName, 'green')
