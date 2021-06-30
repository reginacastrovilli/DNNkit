# this script just takes the input root files and converts them into pkl
# using deprecated uproot3 instead of uproot, as we are not needing high performance from this package, just using it to convert the data
# please neglect warning messages caused by uproot3

import uproot3
import configparser
import ast
from Functions import checkCreateDir
from argparse import ArgumentParser
from colorama import init, Fore
init(autoreset = True)

parser = ArgumentParser()
parser.add_argument('-j', '--JetCollection', help = 'Jet collection: \'TCC\'', type = str, default = 'TCC')
args = parser.parse_args()
jetCollection = args.JetCollection
if args.JetCollection is None:
    parser.error(Fore.RED + 'Requested jet collection (\'TCC\' or )')
elif args.JetCollection != 'TCC':
    parser.error(Fore.RED + 'Jet collection can be \'TCC\', ')

config = configparser.ConfigParser()
config.read('Configuration.ini')
inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
dfPath = config.get('config', 'dfPath')
dfPath += jetCollection + '/'
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
