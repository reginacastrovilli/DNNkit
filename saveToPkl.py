# this script just takes the input root files and converts them into pkl
# using deprecated uproot3 instead of uproot, as we are not needing high performance from this package, just using it to convert the data
# please neglect warning messages caused by uproot3

import uproot3
import configparser
import ast
from colorama import init, Fore
init(autoreset=True)

config = configparser.ConfigParser()
config.read('Config.txt')
inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))

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
    outFile = config.get('config', 'dfPath') + i + '_DF.pkl'
    DF.to_pickle(outFile)
    print('Written ' + outFile)
