Code to produce the (parametric) Deep Neural Network ((P)DNN) for the HDBS analysis.

# Configuration_JETCOLLECTION.txt
A different configuration file for each jet collection is stored.
This text file contains the configuration parameters:
- ntuplePath: path to the flat ntuples produced by the CxAODReader
- dfPath: path to the pandas data frames into which the flat ntuples are converted by the saveToPkl.py file. At each step different subdirectories will be automatically created (if not already existing) where the results will be saved. The subdirectories will be created according to the type of data (jet collection, signal, background, analysis, channel) and neural network (DNN or PDNN) considered.
- inputFiles: list of names of the input .root files that will be converted into pandas dataframes (without absolute path and extension)
- dataType: list of flags that specifies the nature of the corresponding input file (it can be either 'data', 'sig' or 'bkg')
- signals: list of the different signals in the input files
- backgrounds: list of the different backgrounds in the input files
- rootBranchSubSample = list of the variables (root branches) that will actually be converted into pandas dataframes
- InputFeatures(Merged/Resolved) = list of the variables that will be fed to the (P)DNN in the merged/resolved regime

# DSIDtoMass.txt
This text file contains the map used to convert DSID variables into mass values.

# Functions.py
This file contains useful functions that will be called by the other .py scripts. 

# Step 1) saveToPkl.py
This script takes the input .root files and converts them into .pkl files.
It requires one mandatory flag:
- jetCollection (-j): the name of the jet collection used to produce the .root files (e.g. 'TCC', 'UFO_PFLOW')

# Step 2) buildDataset.py
This script takes the .pkl files created in the previous step. A new signal/background flag ('isSignal') is associated to each one of them (1/0). A new mass value ('mass') is associated to each signal event according to the map stored in DSIDtoMass.txt. The 'mass' value of background events is randomly chosen among those listed in DSIDtoMass.txt. For each event only relevant variables (defined in Configuration.txt) are selected and combined into one dataframe, in which the events are shuffled. 
4 input flags can be specified: 
- jetCollection (-j) 
- analysis (-a): type of analysis ('merged' or 'resolved')
- channel (-c): the channel considered ('ggF' or 'VBF')
- preselection cuts (-p): string that will be translated to python commad to filter the inital dataframes according to it (e.g. 'lep1_pt > 0 and lep1_eta > 0')

Only the first two flags are mandatory.

# Step 3) splitDataset.py
This script splits the dataframe produced at the previous step into one train and one test sample. They are also scaled in such a way that the distribution of each of their variables is approximately a gaussian with mean = 0 and sigma = 1 (if possible). It is also possible to consider only a subset of the signal and background types specified in the configuration file.
The following input flags must be specified:
- jetCollection (-j) (mandatory)
- analysis (-a) (mandatory)
- channel (-c) (mandatory)
- preselectionCuts (-p) (optional)
- background (-b): the type of background to select (mandatory, default = value can be found in the ReadArgParser function in Functions.py). Multiple backgrounds can be selected and must be specified in quotation marks separated by a space (e.g. 'Zjets Diboson'). If 'all', all backgrounds in the backgrounds list in the configuration file will be selected
- signal (-s): the signal to select (mandatory)
- training fraction (-t): relative size of the training sample, between 0 and 1 (mandatory, default value can be found in the ReadArgParser function in Functions.py)

# Step 4) buildDNN.py / buildPDNN.py
These scripts run the (Parametric) Deep Neural Network. 
8 input flags can be specified:
- jetCollection (-j)
- analysis (-a)
- channel (-c)
- nodes (-n): number of nodes of the (P)DNN
- layers (-l): number of layers of the (P)DNN
- epochs (-e): number of epochs for the training 
- validation (-v): fraction of the training data that will actually be used for validation
- dropout (-d): fraction of the neurons that will be dropped during the training
- training fraction (-t): relative size of the training sample, between 0 and 1

All these flags are mandatory but only the first 3 must be specficied by the user (the others can also assume their default values).

# Step 5) 
At this point you might want to use your (p)DNN into the CxAODReader. In order to do so, you first have to convert the model into a format that can be read by the reader. The interface between the neural network and the reader is the lwtnn package: https://github.com/lwtnn/lwtnn. This latter comes together with a bunch of conversion scripts. IMPORTANT NOTICE: the CxAODReader also has a duplicate of the lwtnn package. However, as of 2021-05-27, the lwtnn used by the reader is a deprecated version. For converting the model, please clone the latest version of lwtnn from its git repository. Detailed instructions on how to use the conversion tools are found in the README thereof. In summary, as we use a sequential model, we need to run the following command:

  $ python lwtnn/converters/keras2json.py architecture.json variables.json weights.h5 > neural_net.json