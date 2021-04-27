from argparse import ArgumentParser
import configparser, ast

plot = True

parser = ArgumentParser()
parser.add_argument('-t', '--training', help = 'relative size of the training sample, between 0 and 1', default = 0.7)
parser.add_argument('-n', '--nodes', help = 'number of nodes of the DNN, should always be >= nColumns and strictly positive', default = 32)
parser.add_argument('-l', '--layers', help = 'number of layers of the DNN', default = 2)
parser.add_argument('-e', '--epochs', help = 'number of epochs for the training', default = 150)
parser.add_argument('-v', '--validation', help = 'fraction of the training data that will actually be used for validation', default = 0.2)
parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)

args = parser.parse_args()

print('  training =', args.training)
print('     nodes =', args.nodes)
print('    layers =', args.layers)
print('    epochs =', args.epochs)
print('validation =', args.validation)

trainingFraction = float(args.training)
if args.training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error('Training fraction must be between 0 and 1')
numberOfNodes = int(args.nodes)
if args.nodes and numberOfNodes < 1:
    parser.error('Number of nodes must be strictly positive')
numberOfLayers = int(args.layers)
if args.layers and numberOfLayers < 1:
    parser.error('Number of layers must be strictly positive')
numberOfEpochs = int(args.epochs)
if args.epochs and numberOfEpochs < 1:
    parser.error('Number of epochs must be strictly positive')
validationFraction = float(args.validation)
if args.validation and (validationFraction < 0. or validationFraction > 1.):
    parser.error('Validation fraction must be between 0 and 1')
analysis = args.Analysis
if args.Analysis is None:
    parser.error('Requested type of analysis (either \'mergered\' or \'resolved\')')
elif args.Analysis != 'resolved' and args.Analysis != 'merged':
    parser.error('Analysis can be either \'merged\' or \'resolved\'')
channel = args.Channel
if args.Channel is None:
    parser.error('Requested channel (either \'ggF\' or \'VBF\')')
elif args.Channel != 'ggF' and args.Channel != 'VBF':
    parser.error('Channel can be either \'ggF\' or \'VBF\'')

### Reading from config file                                               
config = configparser.ConfigParser()
config.read('Configuration.txt')
dfPath = config.get('config', 'dfPath')
modelPath = config.get('config', 'modelPath')
if analysis == 'merged':
    InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesMerged'))
elif analysis == 'resolved':
    InputFeatures = ast.literal_eval(config.get('config', 'inputFeaturesResolved'))

### Loading data
import pandas as pd
import numpy as np

df_Train = pd.read_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_Train.pkl')
df_Test = pd.read_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_Test.pkl')

### Creating input arrays
X_train = df_Train[InputFeatures].values
y_train_tmp = df_Train['isSignal']
w_train = df_Train['weight'] #unused

X_test = df_Test[InputFeatures].values
y_test_tmp = df_Test['isSignal']
w_test = df_Test['weight'] #unused

### Scaling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

transformerTrain = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, X_train.shape[1] - 1))], remainder = 'passthrough') 
X_train = transformerTrain.fit_transform(X_train)

transformerTest = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, X_test.shape[1] - 1))], remainder = 'passthrough')
X_test = transformerTest.fit_transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train_tmp)
y_test = le.fit_transform(y_test_tmp)

### Building the DNN
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation

def BuildDNN(N_input, width, depth):
    model = Sequential()
    model.add(Dense(units = width, input_dim = N_input))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for i in range(0, depth):
        model.add(Dense(width))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

n_dim = X_train.shape[1]

model = BuildDNN(n_dim, numberOfNodes, numberOfLayers)

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### Training
callbacks = [EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')]

modelMetricsHistory = model.fit(X_train, y_train, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose=1, callbacks=callbacks)

### Evaluating the performance of the DNN
perf = model.evaluate(X_test, y_test, batch_size = 2048)
testLoss = perf[0]
testAccuracy = perf[1]
print('Test loss: ', testLoss)
print('Test  accuracy: ', testAccuracy)

### Drawing training history
import matplotlib
import matplotlib.pyplot as plt

if plot:
    plt.plot(modelMetricsHistory.history['accuracy'])
    plt.plot(modelMetricsHistory.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'lower right')
    plt.figtext(0.5, 0.5, testAccuracy, wrap = True, horizontalalignment = 'center', fontsize = 10)
#    plt.show()
    plt.savefig(modelPath + 'Accuracy.png')
    plt.clf()
    
    plt.plot(modelMetricsHistory.history['loss'])
    plt.plot(modelMetricsHistory.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper right')
    plt.figtext(0.5, 0.5, testLoss, wrap = True, horizontalalignment = 'center', fontsize = 10)
#    plt.show()
    plt.savefig(modelPath + 'Loss.png')
    plt.clf()

### Saving the model
fileName = modelPath + 'Higgs_t' + str(trainingFraction) + '_n' + str(numberOfNodes) + '_l' + str(numberOfLayers) + '_e' + str(numberOfEpochs) + '_v' + str(validationFraction)
model_yaml = model.to_yaml()
with open(fileName + '.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)

model.save_weights(fileName + '.h5')

### Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,normalize = False, title = 'Confusion matrix', cmap=plt.cm.Blues, name = 'CM.png'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
        
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    plt.savefig(plotName)
    #plt.show()
    plt.clf()


### Prediction on the full test sample
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

X_test_signal = X_test[y_test == 1]
m_test = []
for event in X_test_signal:
    m_test.append(event[-1])
massPointsList = list(set(m_test))
print(massPointsList)
X_test_background = X_test[y_test != 1]
for mass in massPointsList:
    X_test_signal_mass = X_test_signal[m_test == mass]
    for bkg in X_test_background:
        bkg[0] = mass 
    X_test_mass = np.concatenate((X_test_signal_mass, X_test_background), axis = 0)
    y_test_signal = np.ones(len(X_test_signal_mass))
    y_test_bkg = np.zeros(len(X_test_background))
    y_test = np.concatenate((y_test_signal, y_test_bkg), axis = 0)
    yhat_test = model.predict(X_test_mass, batch_size = 2048)
    #print(yhat_test)

    ### Plotting ROC
    fpr, tpr, thresholds = roc_curve(y_test, yhat_test)
    roc_auc = auc(fpr, tpr)
    print('ROC_AUC: ', roc_auc)

    if plot:
        realMass = 0
        plt.plot(fpr,  tpr, color = 'darkorange', lw = 2, label = 'Full curve')
        plt.plot([0, 0], [1, 1], color = 'navy', lw = 2, linestyle = '--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC Curves (mass hypothesis: ' + str(int(mass)) + ')')
        plt.legend(loc = 'lower right')
        #    plt.show()
        pltName = modelPath + analysis + '_' + channel + '_ROC_' + str(int(mass)) + '.png'
        plt.savefig(pltName)
        print('Saved ' + pltName)
        plt.clf()

    ### Saving performance parameters
    file = open(fileName + '_' + str(int(mass)) + '.txt', 'w')
    file.write(str(perf[0]))
    file.write(' ')
    file.write(str(perf[1]))
    file.write(' ')
    file.write(str(roc_auc))
    file.close()

    ### Prediction on signal and background separately
    X_train_signal = X_train[y_train == 1]
    X_train_background = X_train[y_train != 1]
    print('Running model prediction on X_train_signal_mass')
    m_train = []
    for trainEvent in X_train_signal:
        m_train.append(trainEvent[-1])
    X_train_signal_mass = X_train_signal[m_train == mass]
    yhat_train_signal = model.predict(X_train_signal_mass, batch_size = 2048)
    print('Running model prediction on X_train_background')
    for bkg in X_train_background:
        bkg[0] = mass
    yhat_train_background = model.predict(X_train_background, batch_size = 2048)
    print('Running model prediction on X_test_signal_mass')
    yhat_test_signal = model.predict(X_test_signal_mass, batch_size = 2048)
    print('Running model prediction on X_test_background')
    yhat_test_background = model.predict(X_test_background, batch_size = 2048)

    ### Plotting scores
    import numpy as np
    if plot:
        bins = np.linspace(0, 1, 40)
        plt.hist(yhat_train_signal, bins = bins, histtype = 'step', lw = 2, color = 'blue', label = [r'Signal Train'], density = True)
        plt.hist(yhat_test_signal, bins = bins, histtype = 'stepfilled', lw = 2, color = 'cyan', alpha = 0.5, label = [r'Signal Test'], density = True)
        plt.hist(yhat_train_background, bins = bins, histtype = 'step', lw = 2, color = 'red', label = [r'Background Train'], density = True)
        plt.hist(yhat_test_background, bins = bins, histtype = 'stepfilled', lw = 2, color = 'orange', alpha = 0.5, label = [r'Background Test'], density = True)
        plt.ylabel('Norm. Entries')
        plt.xlabel('DNN score')
        plt.title('Scores (mass hypothesis: ' + str(int(mass)) + ')')
        plt.legend(loc = 'upper center')
        #    plt.show()
        pltName = modelPath + analysis + '_' + channel + '_Scores_' + str(int(mass)) + '.png'
        plt.savefig(pltName)
        plt.clf()
        print('Saved ' + pltName)

        ### Plotting confusion matrix
        yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test]) #??
        theClasses = ['Backgound','Signal']
        cnf_matrix = confusion_matrix(y_test, yResult_test_cls) #??
        np.set_printoptions(precision = 2)
        titleCM = 'Confusion matrix (Mass hypothesis: ' + str(int(mass)) + ')'
        plotName = modelPath + analysis + '_' + channel + '_ConfusionMatrix_' + str(int(mass)) + '.png'
        plot_confusion_matrix(cnf_matrix, classes = theClasses, normalize = True, title = titleCM, name = plotName)
        print('Saved ' + plotName)
