# this script just takes the input root files and converts them into pkl
# using deprecated uproot3 instead of uproot, as we are not needing high performance from this package, just using it to convert the data
# please neglect warning messages caused by uproot3

import uproot3
import configparser
import ast
import os
from colorama import init, Fore
init(autoreset = True)

def checkCreateDir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        return Fore.RED + ' : created'
    else:
        return Fore.RED + ' : already there'

config = configparser.ConfigParser()
config.read('Configuration.txt')
inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
dfPath = config.get('config', 'dfPath')
print (format('Output directory: ' + Fore.GREEN + dfPath), checkCreateDir(dfPath))

for i in inputFiles:
    inFile = config.get('config', 'ntuplePath') + i + '.root'
    print('Loading ' + inFile)
    theFile = uproot3.open(inFile)
    tree = theFile['Nominal']
    Nevents = tree.numentries
    print('Number of events in ' + inFile, '\t' + str(Nevents))
    if Nevents == 0:
        print(format(Fore.RED + 'Ignoring empty file'))
        continue
    DF = tree.pandas.df()
    outFile = dfPath + i + '_DF.pkl'
    DF.to_pickle(outFile)
    print('Saved ' + outFile)
