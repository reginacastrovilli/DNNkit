### todo list
# 0) place common variables in a config file
# 1) make sure #events is the same in signal and background. This can be done by cutting or by weighting
# DONE 2) select variables to use
# DONE 3) DSID-mass map
# DONE 4) signal flag
# 5) apply preselection cuts

import pandas as pd
import numpy as np

dfPath = '/nfs/kloe/einstein4/HDBS/DNN_InputDataFrames/'
inputFiles = ['Diboson-0', 'Diboson-1', 'Signal']#, 'stop-0', 'stop-1', 'ttbar-0', 'ttbar-1', 'ttbar-2', 'ttbar-3', 'ttbar-4', 'ttbar-5', 'Wjets-0', 'Wjets-1', 'Wjets-2', 'Wjets-3', 'Wjets-4', 'Wjets-5', 'Wjets-6', 'Wjets-7', 'Wjets-8', 'Wjets-9', 'Zjets-0', 'Zjets-10', 'Zjets-11', 'Zjets-1', 'Zjets-2', 'Zjets-3', 'Zjets-4', 'Zjets-5', 'Zjets-6', 'Zjets-7', 'Zjets-8', 'Zjets-9']
dataType   = ['bkg'      , 'bkg'      , 'sig']#   , 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg']
rootBranchSubSample = ['DSID','lep1_m', 'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep2_m','lep2_pt', 'lep2_eta', 'lep2_phi', 'fatjet_m', 'fatjet_pt', 'fatjet_eta', 'fatjet_phi', 'fatjet_D2', 'NJets', 'weight', 'X_boosted_m', 'Zcand_m', 'Zcand_pt', 'Pass_MergHP_GGF_ZZ_Tag_SR', 'Pass_MergHP_GGF_ZZ_Untag_SR', 'Pass_MergLP_GGF_ZZ_Tag_SR', 'Pass_MergLP_GGF_ZZ_Untag_SR', 'Pass_MergHP_GGF_ZZ_Tag_ZCR', 'Pass_MergHP_GGF_ZZ_Untag_ZCR', 'Pass_MergLP_GGF_ZZ_Tag_ZCR', 'Pass_MergLP_GGF_ZZ_Untag_ZCR']


if len(inputFiles) != len(dataType):
    print('Data type array doesnt match input files array')
    exit()

### Loading DSID-mass map
f = open('data/DSIDMap_2lep.txt')
lines = f.readlines()
DSID = [i.split(':')[0] for i in lines]
print (DSID)
mass = [int(i.split(':')[1]) for i in lines]
print (mass)

### Loading pkl files, select only relevant variables, create sig/bkg flag, convert DSID into mass
df = []
counter=0
for i in inputFiles:
#    if (dataType[counter] == 'data'):
    inFile = dfPath+i+'_DF.pkl'
    print('Loading '+inFile)
    newDf=pd.read_pickle(inFile)
    newDf=newDf[rootBranchSubSample]
    if (dataType[counter] == 'sig'):
        newDf.insert(len(newDf.columns), "isSignal", np.ones(newDf.shape[0]), True)
        for k in range(newDf.shape[0]):
            found = False
            for j in range(len(DSID)):
                if (newDf.iat[k,0] == int(DSID[j])):
                    newDf.iat[k,0] = int(mass[j])
                    found = True
            if (found == False):
                print('WARNING !!! missing mass value for DSID ' + str(newDf.iat[k,0]))
    else:
        newDf.insert(len(newDf.columns), "isSignal", np.zeros(newDf.shape[0]), True)
    print(newDf[0:10])
    df.append(newDf)
    counter+=1

### Shuffling data
import sklearn.utils
def Shuffling(df):
    df = sklearn.utils.shuffle(df, random_state=123)
    df = df.reset_index(drop=True)
    return df

for i in range(len(df)):
    print("Shuffling "+inputFiles[i])
    df[i] = Shuffling(df[i])

### divide sig from bkg and select only the number of events needed
df_sig = pd.DataFrame()
df_bkg = pd.DataFrame()
for i in range(len(df)):
    if dataType[i]=='sig':
        df_sig = pd.concat([df_sig, df[i][0:5000]], ignore_index=True) # here change #events signal (todo list 1))
    else:
        df_bkg = pd.concat([df_bkg, df[i][0:1700]], ignore_index=True) # here change #events background (todo list 1))

print(df_sig[0:10])
print(df_bkg[0:10])

print('    Signal events = ', df_sig.shape)
print('Background events = ', df_bkg.shape)

df_total = pd.concat([df_sig, df_bkg], ignore_index=True)
df_total = Shuffling(df_total)
print("Total events after shuffling = ", df_total.shape)

df_total.to_pickle(dfPath+'MixData_PD.pkl')
print('Saved to '+dfPath+'MixData_PD.pkl')
