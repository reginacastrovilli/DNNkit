### todo list
# 0) place common variables in a config file
# 1) make sure #events is the same in signal and background. This can be done by cutting or by weighting
# 2) select variables to use
# 3) DSID-mass map
# 4) signal flag

import pandas as pd

dfPath = '/nfs/kloe/einstein4/HDBS/DNN_InputDataFrames/'
inputFiles = ['Data', 'Diboson-0', 'Diboson-1', 'Signal', 'stop-0', 'stop-1', 'ttbar-0', 'ttbar-1', 'ttbar-2', 'ttbar-3', 'ttbar-4', 'ttbar-5', 'Wjets-0', 'Wjets-1', 'Wjets-2', 'Wjets-3', 'Wjets-4', 'Wjets-5', 'Wjets-6', 'Wjets-7', 'Wjets-8', 'Wjets-9', 'Zjets-0', 'Zjets-10', 'Zjets-11', 'Zjets-1', 'Zjets-2', 'Zjets-3', 'Zjets-4', 'Zjets-5', 'Zjets-6', 'Zjets-7', 'Zjets-8', 'Zjets-9']
dataType   = ['data', 'bkg'      , 'bkg'      , 'sig'   , 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg', 'bkg']
### here put the list of selected variables

if len(inputFiles) != len(dataType):
    print('Data type array doesnt match input files array')
    exit()

### somewhere (perhaps here) select variables (todo list 2))

df = []
for i in inputFiles:
    inFile = dfPath+i+'_DF.pkl'
    print(inFile)
    df.append(pd.read_pickle(inFile))

import sklearn.utils
def Shuffling(df):
    df = sklearn.utils.shuffle(df, random_state=123)
    df = df.reset_index(drop=True)
    return df

for i in range(len(df)):
    print("Shuffling ", inputFiles[i])
    df[i] = Shuffling(df[i])

df_sig = pd.DataFrame()
df_bkg = pd.DataFrame()
for i in range(len(df)):
    if dataType[i]=='sig':
        df_sig = pd.concat([df_sig, df[i][0:5000]], ignore_index=True) # here change #events signal (todo list 1))
    else:
        df_bkg = pd.concat([df_bkg, df[i][0:1700]], ignore_index=True) # here change #events background (todo list 1))

### here add DSID-mass map (todo list 3))

### here add sig/bkg flag (todo list 4))

print('    Signal events = ', df_sig.shape)
print('Background events = ', df_bkg.shape)

df_total = pd.concat([df_sig, df_bkg], ignore_index=True)
df_total = Shuffling(df_total)
print("Total events after shuffling = ", df_total.shape)

df_total.to_pickle(dfPath+'MixData_PD.pkl')

