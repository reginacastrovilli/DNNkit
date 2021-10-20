#import data_preprocess_functions as dp_f
import argparse, configparser
import re
import ast
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import makeplots as mkplots

def find(str_jets,df):
    n=0
    for i in df['origin']==str_jets:
        if i==True:
            n+=1
    return n
    
    
def composition_plot(df):
    x=np.array([])
    samples=list(set(df['origin']))
    for var in samples:
        x=np.append(x,find(var,df))

    plt.figure(1,figsize=(18,6))
    plt.bar(samples,x)
    #plt.text(3, 80, analysis+' '+channel , fontsize=12,horizontalalignment='center',verticalalignment='center')
#    plt.suptitle(analysis+' '+channel+' composition')
    plt.yscale('log')
#    plt.savefig(path+directory+'/'+analysis+'_'+channel+'_'+df_str+'_composition.pdf')
    plt.show()
    return x,samples

analysis='merged'
channel='ggF'
PreselectionCuts=''
samples=['Zjets','Wjets','stop','Diboson','ttbar','Radion','VBFRSG','RSG','VBFRadion','VBFHVTWZ']

### Reading from config file
config = configparser.ConfigParser()
config.read('Configuration.ini')
inputFiles = ast.literal_eval(config.get('config', 'inputFiles'))
dataType = ast.literal_eval(config.get('config', 'dataType'))
rootBranchSubSample = ast.literal_eval(config.get('config', 'rootBranchSubSample'))
dfPath = config.get('config', 'dfPath')

#list of files under dfPath
files=os.listdir(dfPath)
#InputFeaturesResolved for 'resolved' analysis --> make it dyn
if analysis == 'merged':
    dataVariables = ast.literal_eval(config.get('config', 'InputFeaturesMerged'))
elif analysis == 'resolved':
    dataVariables = ast.literal_eval(config.get('config', 'InputFeaturesResolved'))

#It's important the order here ! 
dataVariables.append('isSignal')
dataVariables.append('origin')
dfPath = config.get('config', 'dfPath')

data=pd.read_pickle(dfPath+'MixData_PD_'+analysis+'_'+channel+'.pkl')
data=data[dataVariables]

from sklearn.utils import shuffle
data = shuffle(data)

#select signal
#import os
outdir=dfPath+'outDF/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    print('created: ',outdir)
else:
    print(outdir,'already exists')

for i in range(1,6):
    signal=samples[-i]
#    print(signal)
    training_samples=samples[:5]
    training_samples.append(signal)
    print(training_samples)
    q_string=''
    for k in range(0,len(training_samples)):
        q_string+='origin.str.match("'+training_samples[k]+'") or '
    q_string=q_string[:-4]
#    print(q_string)
    data_set=data.query(q_string, engine='python')
#    data_set=data_set.drop('origin',1)
    data_set.to_pickle(outdir+ signal + 'Data_' + analysis + '_' + channel + '_' + PreselectionCuts+'.pkl')


def LoadData(dfPath, signal, analysis, channel, PreselectionCuts):
    dfInput = dfPath + 'outDF/' + signal + 'Data_' + analysis + '_' + channel + '_' + PreselectionCuts+'.pkl'
    df = pd.read_pickle(dfInput)
    columns=df.columns
    X = df.values
    y = df['isSignal']
    return X, y, columns, df

X,y,columns,df=LoadData(dfPath, signal, analysis, channel, PreselectionCuts)

composition=composition_plot(df)

q_v=[0.3,0.28,0.10,0.3,0.015]
q_v.append(1-sum(q_v))
q=np.array(q_v)
print(np.sum(q))


N_entries=sum(composition[0])
p=np.array(composition[0]/N_entries)

N_v=p[np.argsort(p)]/q[np.argsort(p)]
N_v=N_v[np.argsort(N_v)]
N_prime_tot=int(N_entries*N_v[0])
composition_q=[int(q[n]*N_prime_tot) for n in range(0,len(p))]

q=np.array(composition_q)/N_prime_tot
if (composition_q<=N_entries).all():
#    print('True')
    print(N_prime_tot)
    print(N_prime_tot/N_entries)
    print(q)
    print(composition_q)

training_samples=list(set(df['origin']))
reweighted_df=pd.concat(df.query('origin.str.match("'+training_samples[k]+'")').sample(frac=1)[:composition_q[k]] for k in range(0,len(training_samples)))
reweighted_df_composition=composition_plot(reweighted_df)
#reweighted_df.to_pickle(outdir+ signal + 'Data_' + analysis + '_' + channel + '_' + PreselectionCuts+'_reweighted_dataset.pkl')

test_frac=0.2
X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)

#scaler_train = RobustScaler().fit(X_train[:,:-1])
scaler_train = StandardScaler().fit(X_train[:,:-1])
X_train_scaled=np.array(Sscaler_train.transform(X_train[:,:-1]), dtype=object)
X_test_scaled=np.array(Sscaler_train.transform(X_test[:,:-1]), dtype=object)

y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)

print('train: ',X_train_scaled.mean(),X_train_scaled.std())
print('test: ',X_test_scaled.mean(),X_test_scaled.std())

X_train_scaled_m=np.insert(X_train_scaled, X_train_scaled.shape[1], X_train[:,-1], axis=1)
X_test_scaled_m=np.insert(X_test_scaled, X_test_scaled.shape[1], X_test[:,-1], axis=1)

train_signal_mass=X_train[np.where(X_train[:,-2]==1),-3]
test_signal_mass=X_test[np.where(X_test[:,-2]==1),-3]

b=50
fig, ax = plt.subplots(1,figsize=(15,4))
im1 = ax.hist(train_signal_mass[0],bins=b,alpha=0.6,density=True,label='train_signal')
im2 = ax.hist(test_signal_mass[0],bins=b,alpha=0.6,density=True,label='test_signal')
ax.set_xlabel('mass')
ax.set_xticks(np.sort(np.array(list(set(list(train_signal_mass.flatten()))))))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.legend()
plt.show()






