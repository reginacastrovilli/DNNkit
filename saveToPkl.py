# this script just takes the input root files and converts them into pkl
# using deprecated uproot3 instead of uproot, as we are not needing high performance from this package, just using it to convert the data
# please neglect warning messages caused by uproot3

import uproot3
import configparser
import ast
from Functions import ReadArgParser, ReadConfigSaveToPkl
from colorama import init, Fore
init(autoreset = True)

### Reading the command line
jetCollection = ReadArgParser()

### Reading from config file
ntuplePath, inputFiles, dfPath = ReadConfigSaveToPkl(jetCollection)

### Loading, converting and saving each input file
for i in inputFiles:
    inFile = ntuplePath + i + '.root'
    print('Loading ' + inFile)
    theFile = uproot3.open(inFile)
    tree = theFile['Nominal']
    Nevents = tree.numentries
    print('Number of events in ' + inFile, '\t' + str(Nevents))
    if Nevents == 0:
        print(Fore.RED + 'Ignoring empty file')
        continue
    DF = tree.pandas.df()
    outFile = dfPath + i + '_DF.pkl'
    DF.to_pickle(outFile)
    print('Saved ' + outFile)
