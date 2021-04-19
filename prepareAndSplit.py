from argparse import ArgumentParser

plot=True

parser = ArgumentParser()
parser.add_argument("-t", "--training", help="relative size of the training sample, between 0 and 1", default = 0.7)

args=parser.parse_args()

print('  training =', args.training)

trainingFraction = float(args.training)
if args.training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error("training fraction must be between 0 and 1")

dfPath = '/nfs/kloe/einstein4/HDBS/DNN_InputDataFrames/'

################################################################
####### loading data

import numpy as np
import pandas as pd

df_total=pd.read_pickle(dfPath+'MixData_PD.pkl')

################################################################
####### splitting sample

Ntrain_stop = int(round(df_total.shape[0] * trainingFraction))
X_Train = df_total[:Ntrain_stop]
X_Test = df_total[Ntrain_stop:]
X_Train.to_pickle(dfPath+'MixData_PD_Train.pkl')
print('Saved '+dfPath+'MixData_PD_Train.pkl')
print(X_Train[:10])
X_Test.to_pickle(dfPath+'MixData_PD_Test.pkl')
print('Saved '+dfPath+'MixData_PD_Test.pkl')
print(X_Test[:10])

print('Size of the training saple = ', X_Train.shape)
print('Size of the testing saple = ', X_Test.shape)
