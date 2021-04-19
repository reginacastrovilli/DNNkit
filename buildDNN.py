from argparse import ArgumentParser

plot=True

parser = ArgumentParser()
parser.add_argument("-t", "--training", help="relative size of the training sample, between 0 and 1", default = 0.7)
parser.add_argument("-n", "--nodes", help="number of nodes of the DNN, should always be >= nColumns and strictly positive", default = 32)
parser.add_argument("-l", "--layers", help="number of layers of the DNN", default = 2)
parser.add_argument("-e", "--epochs", help="number of epochs for the training", default = 150)
parser.add_argument("-v", "--validation", help="fraction of the training data that will actually be used for validation", default = 0.2)

args=parser.parse_args()

print('  training =', args.training)
print('     nodes =', args.nodes)
print('    layers =', args.layers)
print('    epochs =', args.epochs)
print('validation =', args.validation)

trainingFraction = float(args.training)
if args.training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error("training fraction must be between 0 and 1")
numberOfNodes = int(args.nodes)
if args.nodes and numberOfNodes < 1:
    parser.error("number of nodes must be strictly positive")
numberOfLayers = int(args.layers)
if args.layers and numberOfLayers < 1:
    parser.error("number of layers must be strictly positive")
numberOfEpochs = int(args.epochs)
if args.epochs and numberOfEpochs < 1:
    parser.error("number of epochs must be strictly positive")
validationFraction = float(args.validation)
if args.validation and (validationFraction < 0. or validationFraction > 1.):
    parser.error("validation fraction must be between 0 and 1")

dfPath = '/nfs/kloe/einstein4/HDBS/DNN_InputDataFrames/'
modelPath = '/nfs/kloe/einstein4/HDBS/DNNModels/'

InputFeatures = ['DSID','lep1_m', 'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep2_m','lep2_pt', 'lep2_eta', 'lep2_phi', 'fatjet_m', 'fatjet_pt', 'fatjet_eta', 'fatjet_phi', 'fatjet_D2', 'NJets', 'X_boosted_m', 'Zcand_m', 'Zcand_pt']

################################################################
####### loading data

import numpy as np
import pandas as pd

df_Train=pd.read_pickle(dfPath+'MixData_PD_Train.pkl')
df_Test=pd.read_pickle(dfPath+'MixData_PD_Test.pkl')

################################################################
####### creating input arrays

X_train=df_Train[InputFeatures].values
y_train_tmp = df_Train['isSignal']
w_train=df_Train['weight']

X_test=df_Test[InputFeatures].values
y_test_tmp = df_Test['isSignal']
w_test=df_Test['weight']

################################################################
#######  scaling

from sklearn.preprocessing import StandardScaler,LabelEncoder

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train_tmp)
y_test = le.fit_transform(y_test_tmp)

################################################################
####### building the DNN

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation

def BuildDNN(N_input, width, depth):
    model = Sequential()
    model.add(Dense(units=width, input_dim=N_input))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for i in range(0, depth):
        model.add(Dense(width))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

n_dim=X_train.shape[1]

model=BuildDNN(n_dim, numberOfNodes, numberOfLayers)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

################################################################
####### training

callbacks = [EarlyStopping(verbose=True, patience=10, monitor='val_loss')]

modelMetricsHistory = model.fit(X_train, y_train, epochs=numberOfEpochs, batch_size=2048, validation_split=validationFraction, verbose=1, callbacks=callbacks)

################################################################
####### evaluating the performance of the DNN

perf=model.evaluate(X_test, y_test, batch_size=2048)
testLoss = 'Test loss:', perf[0]
testAccuracy = 'Test accuracy:', perf[1]
print(testLoss)
print(testAccuracy)

################################################################
####### drawing training history

import matplotlib
import matplotlib.pyplot as plt

if plot:
    plt.plot(modelMetricsHistory.history['accuracy'])
    plt.plot(modelMetricsHistory.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.figtext(0.5, 0.5, testAccuracy, wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()
    plt.savefig(modelPath+"Accuracy.png")
    plt.clf()
    
    plt.plot(modelMetricsHistory.history['loss'])
    plt.plot(modelMetricsHistory.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.figtext(0.5, 0.5, testLoss, wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()
    plt.savefig(modelPath+"Loss.png")
    plt.clf()

################################################################
####### saving the model

fileName = modelPath+"Higgs_t" + str(trainingFraction) + "_n" + str(numberOfNodes) + "_l" + str(numberOfLayers) + "_e" + str(numberOfEpochs) + "_v" + str(validationFraction)
model_yaml = model.to_yaml()
with open(fileName + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

model.save_weights(fileName + ".h5")

################################################################
####### prediction on the full test sample

from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
yhat_test = model.predict(X_test, batch_size=2048)
print(yhat_test)

################################################################
####### plotting ROC

fpr, tpr, thresholds = roc_curve(y_test, yhat_test)
roc_auc = auc(fpr, tpr)
print("ROC_AUC: %0.3f", roc_auc)

if plot:
    plt.plot(fpr,  tpr, color='darkorange', lw=2, label='Full curve')
    plt.plot([0,0], [1,1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(modelPath+"ROC.png")
    plt.clf()

################################################################
####### saving performance parameters

file = open(fileName + ".txt", "w")
file.write(str(perf[0]))
file.write(" ")
file.write(str(perf[1]))
file.write(" ")
file.write(str(roc_auc))
file.close()

################################################################
####### prediction on signal and backgrouund separately

Xtrain_signal = X_train[y_train==1]
Xtrain_background = X_train[y_train!=1]
Xtest_signal = X_test[y_test==1]
Xtest_background = X_test[y_test!=1]

print('Running model prediction on Xtrain_signal')
yhat_train_signal = model.predict(Xtrain_signal, batch_size=2048)
print('Running model prediction on Xtrain_background')
yhat_train_background = model.predict(Xtrain_background, batch_size=2048)
print('Running model prediction on Xtest_signal')
yhat_test_signal = model.predict(Xtest_signal, batch_size=2048)
print('Running model prediction on Xtest_background')
yhat_test_background = model.predict(Xtest_background, batch_size=2048)

################################################################
####### plotting scores

if plot:
    bins=np.linspace(0,1,40)
    plt.hist(yhat_train_signal, bins=bins, histtype='step', lw=2, color='blue', label=[r'Signal Train'], density=True)
    plt.hist(yhat_test_signal, bins=bins, histtype='stepfilled', lw=2, color='cyan', alpha=0.5, label=[r'Signal Test'], density=True)
    plt.hist(yhat_train_background, bins=bins, histtype='step', lw=2, color='red', label=[r'Background Train'], density=True)
    plt.hist(yhat_test_background, bins=bins, histtype='stepfilled', lw=2, color='orange', alpha=0.5, label=[r'Background Test'], density=True)
    
    plt.ylabel('Norm. Entries')
    plt.xlabel('DNN score')
    plt.legend(loc="upper center")
    plt.show()
    plt.savefig(modelPath+"Scores.png")
    plt.clf()
