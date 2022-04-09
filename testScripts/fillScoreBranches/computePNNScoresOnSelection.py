from utilityFunctions import SelectEvents
import uproot3
from colorama import init, Fore
init(autoreset = True)
import json
import numpy as np
from argparse import ArgumentParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} ---> '3' to suppress INFO, WARNING, and ERROR messages in Tensorflow


import matplotlib.pyplot as plt
from datetime import datetime
from time import process_time
import time

        
## starting real and cpu time 
trealstart=time.time()
tcpustart=process_time()
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string_fn = now.strftime("%d_%m_%Y_%H_%M_%S")
print(' ')
print(Fore.RED + 'Starting at ' + dt_string + ' string for outputDir ' + dt_string_fn)


### Reading the command line
parser = ArgumentParser()
parser.add_argument('-i', '--Input', help = 'Input file: \'data18-0\' or \'Zjet-2\'', type = str)
parser.add_argument('-m', '--MCtype', help = 'Monte Carlo type: \'mc16a\' or \'mc16d\' or \'mc16e\'', type = str)
parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
parser.add_argument('-s', '--Signal', help = 'Signal: \'VBFHVTWZ\', \'Radion\', \'RSG\', \'VBFRSG\', \'HVTWZ\' or \'VBFRadion\'', type = str)
parser.add_argument('-b', '--Background', help = 'Background: \'Zjets\', \'Wjets\', \'stop\', \'Diboson\', \'ttbar\' or \'all\' (in quotation mark separated by a space)', type = str, default = 'all')
parser.add_argument('-t', '--TAG',    help = 'Tag for output directory name  (under /scratch/stefania): \'any_string\'', type = str, default = 'tmp')
args = parser.parse_args()
inpF=args.Input
analysis = args.Analysis
channel = args.Channel
signal = args.Signal
background = args.Background
mcType=args.MCtype
dirTag=args.TAG
if inpF.endswith('.root'):
    inpF = inpF[:-5]
print(inpF)
print(Fore.BLUE + 'Arguments are ' + inpF + ' ' + analysis + ' ' + channel + ' ' + signal + ' ' + mcType + ' ' + dirTag)


modelDir = '/nfs/kloe/einstein4/HDBS/NNoutput/r33-22/UFO_PFLOW/' + analysis + '/' + channel + '/' + signal + '/' + background + '_fullStat/PDNN/'
inputFeatures = np.array([])
inputFeaturesAndFlags = np.array([])

### Loading model produced by the NN                                                                                                                              
from keras.models import model_from_json
architectureFile = modelDir + 'architecture.json'
print(Fore.GREEN + 'Architecture File: ' + architectureFile)
if not os.path.isfile(architectureFile):
    print(Fore.RED + 'Architecture File: ' + architectureFile + ' does not exist')
    exit()
with open(architectureFile, 'r') as json_file:
    print(Fore.GREEN + 'Loading ' + architectureFile)
    model = model_from_json(''.join(json_file.readlines()))

### Loading weights into the model
weightsFile = modelDir + 'weights.h5'
model.load_weights(weightsFile)
print(Fore.GREEN + 'Loading ' + weightsFile)

### Loading and storing scaling values 
variablesFile = modelDir + 'variables.json'
jsonFile = open(variablesFile, 'r')
print(Fore.GREEN + 'Loading ' + variablesFile)
values = json.load(jsonFile)
offsetDict = {}
scaleDict = {}
for field in values['inputs']:
    feature = field['name']
    offsetDict[feature] = field['offset']
    scaleDict[feature] = field['scale']
    if feature != 'mass':
        inputFeatures = np.append(inputFeatures, feature)
        inputFeaturesAndFlags = np.append(inputFeaturesAndFlags, feature)
jsonFile.close()
num_variables= np.shape(inputFeatures)[0]
print(Fore.BLUE + 'Number of variables ' + str(num_variables))


passFlags=['Pass_MergHP_GGF_ZZ_Tag_SR', 'Pass_MergHP_GGF_ZZ_Untag_SR', 'Pass_MergLP_GGF_ZZ_Tag_SR', 'Pass_MergLP_GGF_ZZ_Untag_SR', 'Pass_MergHP_GGF_ZZ_Tag_ZCR', 'Pass_MergHP_GGF_ZZ_Untag_ZCR', 'Pass_MergLP_GGF_ZZ_Tag_ZCR', 'Pass_MergLP_GGF_ZZ_Untag_ZCR', 'Pass_MergHP_GGF_WZ_SR', 'Pass_MergHP_GGF_WZ_ZCR', 'Pass_MergLP_GGF_WZ_SR', 'Pass_MergLP_GGF_WZ_ZCR', 'Pass_MergHP_VBF_WZ_SR', 'Pass_MergHP_VBF_ZZ_SR', 'Pass_MergHP_VBF_WZ_ZCR', 'Pass_MergHP_VBF_ZZ_ZCR', 'Pass_MergLP_VBF_WZ_SR', 'Pass_MergLP_VBF_ZZ_SR', 'Pass_MergLP_VBF_WZ_ZCR', 'Pass_MergLP_VBF_ZZ_ZCR', 'Pass_Res_GGF_WZ_SR', 'Pass_Res_GGF_WZ_ZCR', 'Pass_Res_GGF_ZZ_Tag_SR', 'Pass_Res_GGF_ZZ_Untag_SR', 'Pass_Res_GGF_ZZ_Tag_ZCR', 'Pass_Res_GGF_ZZ_Untag_ZCR', 'Pass_Res_VBF_WZ_SR', 'Pass_Res_VBF_WZ_ZCR', 'Pass_Res_VBF_ZZ_SR', 'Pass_Res_VBF_ZZ_ZCR']
num_flags= np.shape(passFlags)[0]
num_flags= num_flags+1
print(Fore.BLUE + 'Number of variables ' + str(num_variables) + ' number of flags ' + str(num_flags))

passFlagsMerg=['Pass_MergHP_GGF_ZZ_Tag_SR', 'Pass_MergHP_GGF_ZZ_Untag_SR', 'Pass_MergLP_GGF_ZZ_Tag_SR', 'Pass_MergLP_GGF_ZZ_Untag_SR', 'Pass_MergHP_GGF_ZZ_Tag_ZCR', 'Pass_MergHP_GGF_ZZ_Untag_ZCR', 'Pass_MergLP_GGF_ZZ_Tag_ZCR', 'Pass_MergLP_GGF_ZZ_Untag_ZCR', 'Pass_MergHP_GGF_WZ_SR', 'Pass_MergHP_GGF_WZ_ZCR', 'Pass_MergLP_GGF_WZ_SR', 'Pass_MergLP_GGF_WZ_ZCR', 'Pass_MergHP_VBF_WZ_SR', 'Pass_MergHP_VBF_ZZ_SR', 'Pass_MergHP_VBF_WZ_ZCR', 'Pass_MergHP_VBF_ZZ_ZCR', 'Pass_MergLP_VBF_WZ_SR', 'Pass_MergLP_VBF_ZZ_SR', 'Pass_MergLP_VBF_WZ_ZCR', 'Pass_MergLP_VBF_ZZ_ZCR']

passFlagsRes=['Pass_Res_GGF_WZ_SR', 'Pass_Res_GGF_WZ_ZCR', 'Pass_Res_GGF_ZZ_Tag_SR', 'Pass_Res_GGF_ZZ_Untag_SR', 'Pass_Res_GGF_ZZ_Tag_ZCR', 'Pass_Res_GGF_ZZ_Untag_ZCR', 'Pass_Res_VBF_WZ_SR', 'Pass_Res_VBF_WZ_ZCR', 'Pass_Res_VBF_ZZ_SR', 'Pass_Res_VBF_ZZ_ZCR']

passFlagsResGGF=['Pass_Res_GGF_WZ_SR', 'Pass_Res_GGF_WZ_ZCR', 'Pass_Res_GGF_ZZ_Tag_SR', 'Pass_Res_GGF_ZZ_Untag_SR', 'Pass_Res_GGF_ZZ_Tag_ZCR', 'Pass_Res_GGF_ZZ_Untag_ZCR']

passFlagsResVBF=['Pass_Res_VBF_WZ_SR', 'Pass_Res_VBF_WZ_ZCR', 'Pass_Res_VBF_ZZ_SR', 'Pass_Res_VBF_ZZ_ZCR']

passFlagsMergGGF=['Pass_MergHP_GGF_ZZ_Tag_SR', 'Pass_MergHP_GGF_ZZ_Untag_SR', 'Pass_MergLP_GGF_ZZ_Tag_SR', 'Pass_MergLP_GGF_ZZ_Untag_SR', 'Pass_MergHP_GGF_ZZ_Tag_ZCR', 'Pass_MergHP_GGF_ZZ_Untag_ZCR', 'Pass_MergLP_GGF_ZZ_Tag_ZCR', 'Pass_MergLP_GGF_ZZ_Untag_ZCR', 'Pass_MergHP_GGF_WZ_SR', 'Pass_MergHP_GGF_WZ_ZCR', 'Pass_MergLP_GGF_WZ_SR', 'Pass_MergLP_GGF_WZ_ZCR']

passFlagsMergVBF=['Pass_MergHP_VBF_WZ_SR', 'Pass_MergHP_VBF_ZZ_SR', 'Pass_MergHP_VBF_WZ_ZCR', 'Pass_MergHP_VBF_ZZ_ZCR', 'Pass_MergLP_VBF_WZ_SR', 'Pass_MergLP_VBF_ZZ_SR', 'Pass_MergLP_VBF_WZ_ZCR', 'Pass_MergLP_VBF_ZZ_ZCR']

for x in np.nditer(passFlags):
    inputFeaturesAndFlags = np.append(inputFeaturesAndFlags, x)
inputFeaturesAndFlags = np.append(inputFeaturesAndFlags, 'Pass_isVBF')
print(Fore.BLUE + 'Number of variables+flags ' + str(np.shape(inputFeaturesAndFlags)[0]))

outputDir = dirTag

### Loading and converting each input file
inputDir = '/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_'+mcType+'_VV_2lep_PFlow_UFO/fetch/data-MVATree/'
#inputFiles = ['data18-0'] ### extend
inputFiles = [inpF] ### extend
print(Fore.BLUE + 'Input file ' + inputDir + inpF + '.root')
scoresDict = {}
massHypo = np.linspace(500, 6500, 61, dtype = np.int64) ### creates an array of 61 int between 500 and 6500

for inputFile in inputFiles:
    t0real=time.time()
    t0cpu=process_time()
    outputFileName = '/Scores_' + inputFile + '_'+analysis+'_' + channel + '_' + signal + '.txt'
    outputFile = open(outputDir + outputFileName, 'w')
    print(' ')
    print(Fore.GREEN + 'Loading ' + inputDir + inputFile + '.root')
    theFile = uproot3.open(inputDir + inputFile + '.root')
    tree = theFile['Nominal']
    Nevents = tree.numentries
    print(Fore.BLUE + 'Number of events in tree: ' + str(Nevents))
    dataFrame = tree.pandas.df(inputFeatures)
    dataFrameWFlags = tree.pandas.df(inputFeaturesAndFlags)
    nrbefore  = np.shape(dataFrameWFlags)[0]
    print(Fore.BLUE + 'Number of raws   in DF  : ' + str(nrbefore))

    ### select a sub-df with events passing at least one selection
    selectedDfWFlags = SelectEvents(dataFrameWFlags, channel, analysis)
    nrafter  = np.shape(selectedDfWFlags)[0]
    ncbefore = np.shape(selectedDfWFlags)[1]
    print(Fore.BLUE + 'Number of raws after sel: ' + str(nrafter) + ' fraction = ' + str(nrafter/nrbefore))
    print(Fore.BLUE + 'Number of cols after sel: ' + str(ncbefore))

    ## nothing to do for this file
    if nrafter == 0:
        print(Fore.BLUE + 'Moving to next file ')
        continue

    
    ### remove coloums of flags (remove from coloumn "num_variables" to the last coloum in step of 1)
    #selectedDf = np.delete(selectedDfWFlags, np.s_[first_col:last_col:step_col], 1=coloumn)
    #selectedDf = np.delete(selectedDfWFlags, np.s_[num_variables:ncbefore:1], 1)
    #selectedDf = selectedDfWFlags[0:nrafter:1, 0:num_variables:1]
    selectedDf = selectedDfWFlags.drop(columns=['Pass_isVBF'], axis=1)
    print (Fore.BLUE + 'Pass_isVBF removed, n.of col. is now '+ str(np.shape(selectedDf)[1]))
    selectedDf = selectedDf.drop(columns=passFlags, axis=1)
    ncafter    = np.shape(selectedDf)[1]
    print(Fore.BLUE + 'Number of variables ' + str(num_variables) + ' number of flags ' + str(num_flags))
    print(Fore.BLUE + 'Number of coloumns after removing flags ' + str(ncafter))
    
    
    ### Scaling input features
    for column in dataFrame.columns:
        #dataFrame[column] = (dataFrame[column] - offsetDict[column]) / scaleDict[column]
        selectedDf[column] = (selectedDf[column] - offsetDict[column]) / scaleDict[column]

    ### Testing different mass hypothesis
    for mass in massHypo:
        print(Fore.BLUE + 'Testing mass hypothesis: ' + str(mass) + ' GeV')
        branchName = 'pDNNScore' + str(mass)
        ### Assigning new mass column
        scaledMass = (mass - offsetDict['mass']) / scaleDict['mass']
        #dataFrame = dataFrame.assign(branchName = np.full(dataFrame.shape[0], scaledMass))
        selectedDf = selectedDf.assign(branchName = np.full(selectedDf.shape[0], scaledMass))
        ### Evaluating scores
        #scores = model.predict(np.array(dataFrame.values).astype(np.float32), batch_size = 2048)
        scores = model.predict(np.array(selectedDf.values).astype(np.float32), batch_size = 2048)
        scoresDict[mass] = scores

    t1real=time.time()
    t1cpu=process_time()
    DTreal = t1real-t0real
    DTcpu  = t1cpu -t0cpu
    print(Fore.GREEN + 'Score computation over for ' + str(nrafter) + ' events in ' + inputFile+'; Real / cpu time spent: ' + format(DTreal, ".2f") + ' / ' + format(DTcpu, ".2f") + ' s; time/event = ' + format(DTreal/nrafter, ".4f") + ' / ' + format(DTcpu/nrafter, ".4f"))


    ### Saving score to file
    #for iEvent in range(dataFrame.shape[0]):
    for iEvent in range(selectedDf.shape[0]):
        for mass in massHypo:
            outputFile.write(str(scoresDict[mass][iEvent][0]) + ' ')
        outputFile.write('\n')

    outputFile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + outputFileName)


    Nbins=100
    plt.hist(scoresDict[600], bins = np.arange(-0.01, 1.01, 0.01), histtype = 'step', color = 'blue', label = [r'pDNNScore600'], density = False)
    MeanScore600 = np.mean(scoresDict[600]);
    RMSScore600  = np.std(scoresDict[600]);
    Entries600   = len(scoresDict[600]);
    plt.ylabel('Entries')
    plt.xlabel('Score600')
    plt.yscale('log')
    plt.title('prova')
    plt.legend(loc = 'upper center')
    ScoresPltName = 'prova.png'
    plt.savefig(ScoresPltName)
    #plt.draw()
    print(Fore.GREEN + 'Saved ' + ScoresPltName)
    print(Fore.RED + 'Entries, mean, rms = ' , Entries600, ' ' , MeanScore600 , ' ' , RMSScore600)
    plt.clf()
    plt.close()

