# this script just takes the input root files and converts them into pkl
# using deprecated uproot3 instead of uproot, as we are not needing high performance from this package, just using it to convert the data
# please neglect warning messages caused by uproot3

import uproot3

ntuplePath = '/nfs/kloe/einstein4/HDBS/newDataFromRob/'
dfPath = '/nfs/kloe/einstein4/HDBS/DNN_InputDataFrames/'
inputFiles = ['Data', 
              'Diboson-0', 
              'Diboson-1', 
              'Signal', 
              'stop-0', 
              'stop-1', 
              'ttbar-0', 
              'ttbar-1', 
              'ttbar-2', 
              'ttbar-3', 
              'ttbar-4', 
              'ttbar-5', 
              'Wjets-0', 
              'Wjets-1', 
              'Wjets-2', 
              'Wjets-3', 
              'Wjets-4', 
              'Wjets-5', 
              'Wjets-6', 
              'Wjets-7', 
              'Wjets-8', 
              'Wjets-9', 
              'Zjets-0', 
              'Zjets-10', 
              'Zjets-11', 
              'Zjets-1', 
              'Zjets-2', 
              'Zjets-3', 
              'Zjets-4', 
              'Zjets-5', 
              'Zjets-6', 
              'Zjets-7', 
              'Zjets-8', 
              'Zjets-9']

for i in inputFiles:
    inFile = ntuplePath+i+'.root'
    print('Loading '+inFile)
    theFile = uproot3.open(inFile)
    tree = theFile['Nominal']
    Nevents = tree.numentries
    print('Number of events in '+inFile,'\t'+str(Nevents))
    DF = tree.pandas.df()
    outFile = dfPath+i+'_DF.pkl'
    DF.to_pickle(outFile)
    print('Written '+outFile)
