Code to produce the (parametric) Deep Neural Network ((p)DNN) for the HDBS analysis.

# Configuration.txt
This text file contains the configuration parameters:
- ntuplePath: path to the flat ntuples produced by the CxAODReader
- dfPath: path to the pandas data frames into which the flat ntuples are converted by the saveToPkl.py file
- modelPath: path to directory in which the results of the (p)DNN will be stored
- inputFiles: list of names of the input .root files that will be converted into pandas dataframes (without absolute path and extension)
- dataType: list of flags that specifies the nature of the corresponding input file (it can be either 'data', 'sig' or 'bkg')
- rootBranchSubSample = list of the variables (root branches) that will actually be converted into pandas dataframes
- InputFeatures(Merged/Resolved) = list of the variables that will be fed to the (p)DNN in the merged/resolved regime

# Functions.py
This file contains useful functions that will be called by the other .py scripts. 

# Step 1) saveToPkl.py
This script takes the input .root files and converts them into .pkl files.

# Step 2) buildDataset.py
This script takes the .pkl files created at the previous step, selects only relevant (user-defined) variables and combines them into one data frame. Events are shuffled in the output data frame and a new signal/background flag is associated to each one of them. 
3 input flags can be specified: 
- analysis (-a): type of analysis ('merged' or 'resolved')
- channel (-c): the channel considered ('ggF' or 'VBF')
- preselection cuts (-p): string that will be translated to python commad to filter the inital PDs according to it (e.g. 'lep1_pt > 0 and lep1_eta > 0')
Only the first two flags are mandatory.

# Step 3) splitDataset.py
This script splits the data frame produced at the previous step into train and test samples and saves them into two separate .pkl files.
3 input flags must be specified:
- analysis (-a): type of analysis ('merged' or 'resolved')
- channel (-c): the channel considered ('ggF' or 'VBF')
- training fraction: relative size of the training sample, between 0 and 1
All these flags are mandatory but only the first two must be specficied by the user (the last one can also assume their default values).

# Step 4) buildDNN.py / buildPDNN.py
These scripts run the (parametric) Deep Neural Network. 
7 input flags can be specified:
- analysis (-a): type of analysis ('merged' or 'resolved')
- channel (-c): the channel considered ('ggF' or 'VBF')
- nodes (-n): number of nodes of the (p)DNN
- layers (-l): number of layers of the (p)DNN
- epochs (-e): number of epochs for the training 
- validation (-v): fraction of the training data that will actually be used for validation
- dropout (-d): fraction of the neurons that will be dropped during the training
All these flags are mandatory but only the first two must be specficied by the user (the other five can also assume their default values).