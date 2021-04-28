from argparse import ArgumentParser
import configparser, ast
from Functions import *

plot = True

parser = ArgumentParser()

parser.add_argument('-a', '--Analysis', help = 'Type of analysis: \'merged\' or \'resolved\'', type = str)
parser.add_argument('-c', '--Channel', help = 'Channel: \'ggF\' or \'VBF\'', type = str)
parser.add_argument('-t', '--Training', help = 'Relative size of the training sample, between 0 and 1', default = 0.7)
parser.add_argument('-n', '--Nodes', help = 'Number of nodes of the DNN, should always be >= nColumns and strictly positive', default = 32)
parser.add_argument('-l', '--Layers', help = 'Number of layers of the DNN', default = 2)
parser.add_argument('-e', '--Epochs', help = 'Number of epochs for the training', default = 150)
parser.add_argument('-v', '--Validation', help = 'Fraction of the training data that will actually be used for validation', default = 0.2)

args = parser.parse_args()

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
trainingFraction = float(args.Training)
if args.Training and (trainingFraction < 0. or trainingFraction > 1.):
    parser.error('Training fraction must be between 0 and 1')
numberOfNodes = int(args.Nodes)
if args.Nodes and numberOfNodes < 1:
    parser.error('Number of nodes must be strictly positive')
numberOfLayers = int(args.Layers)
if args.Layers and numberOfLayers < 1:
    parser.error('Number of layers must be strictly positive')
numberOfEpochs = int(args.Epochs)
if args.Epochs and numberOfEpochs < 1:
    parser.error('Number of epochs must be strictly positive')
validationFraction = float(args.Validation)
if args.Validation and (validationFraction < 0. or validationFraction > 1.):
    parser.error('Validation fraction must be between 0 and 1')

print('  training =', trainingFraction)
print('     nodes =', numberOfNodes)
print('    layers =', numberOfLayers)
print('    epochs =', numberOfEpochs)
print('validation =', validationFraction)

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

df_Train = pd.read_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_Train.pkl')
df_Test = pd.read_pickle(dfPath + 'MixData_PD_' + analysis + '_' + channel + '_Test.pkl')

### Creating input arrays
X_train = df_Train[InputFeatures].values
y_train = df_Train['isSignal']
w_train = df_Train['weight'] 

X_test = df_Test[InputFeatures].values
y_test = df_Test['isSignal']
w_test = df_Test['weight'] 

### Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, X_train.shape[1] - 1))], remainder = 'passthrough') 
X_train = transformer.fit_transform(X_train)
X_test = transformer.fit_transform(X_test)

### Building the DNN
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
print(format(Fore.BLUE + 'Test loss: ' + str(testLoss)))
print(format(Fore.BLUE + 'Test  accuracy: ' + str(testAccuracy)))

### Drawing training history
outputDir = modelPath + 'pDNN/' + analysis + '/' + channel
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

if plot:
    titleAccuracy = 'Model accuracy'
    AccuracyPltName = outputDir + '/Accuracy.png'
    DrawAccuracy(modelMetricsHistory, testAccuracy, titleAccuracy, AccuracyPltName)

    titleLoss = 'Model loss'
    LossPltName = outputDir + '/Loss.png'
    DrawLoss(modelMetricsHistory, testLoss, titleLoss, LossPltName)

### Saving the model
fileName = outputDir + '/Higgs_t' + str(trainingFraction) + '_n' + str(numberOfNodes) + '_l' + str(numberOfLayers) + '_e' + str(numberOfEpochs) + '_v' + str(validationFraction)
model_yaml = model.to_yaml()
with open(fileName + '.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
print('Saved ' + fileName + '.yaml')
model.save_weights(fileName + '.h5')
print('Saved ' + fileName + '.h5')

###### Prediction on the full test sample

### Dividing signal from background
X_test_signal = X_test[y_test == 1]
X_test_bkg = X_test[y_test != 1]

### Saving mass values
m_test_signal = []
for event in X_test_signal:
    m_test_signal.append(event[-1])
massPointsList = list(set(m_test_signal))
print('Mass points: ' + str(massPointsList))

for mass in massPointsList:

    ### Creating the logFile
    logFileName = outputDir + '/logFile.txt'
    logFile = open(logFileName, 'w')

    ### Writing previous information to the logFile
    logFile.write('dfPath: ' + dfPath + '\nmodelPath: ' + modelPath + '\nInputFeatures: ' + str(InputFeatures) + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nTraining fraction: ' + str(trainingFraction) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nMass points list: ' + str(massPointsList) + '\nMass point analyzed: ' + str(mass))

    ### Selecting only signal events with the same mass value
    X_test_signal_mass = X_test_signal[m_test_signal == mass]

    ### Assigning the same mass value to background events
    for bkg in X_test_bkg:
        bkg[0] = mass 

    ### Putting signal and background events back togheter
    X_test_mass = np.concatenate((X_test_signal_mass, X_test_bkg), axis = 0)

    ### Creating y_test_mass with the same dimension of X_test_mass
    y_test_signal_mass = np.ones(len(X_test_signal_mass))
    y_test_bkg = np.zeros(len(X_test_bkg))
    y_test_mass = np.concatenate((y_test_signal_mass, y_test_bkg), axis = 0)

    ### Prediction
    yhat_test_mass = model.predict(X_test_mass, batch_size = 2048)
    #print(yhat_test_mass)

    ### Plotting ROC
    fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)
    roc_auc = auc(fpr, tpr)
    print('ROC_AUC: ', roc_auc)

    if plot:
        ### Creating new output directory for each mass value
        newOutputDir = outputDir + '/' + str(int(mass))
        print (format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))

        titleROC = 'ROC Curves (mass hypotesis: ' + str(int(mass)) + ')'
        ROCPltName = newOutputDir + '/ROC.png'
        DrawROC(fpr, tpr, titleROC, ROCPltName)

        ### Plotting confusion matrix
        yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test_mass])
        cnf_matrix = confusion_matrix(y_test_mass, yResult_test_cls)
        titleCM = 'Confusion matrix (Mass hypothesis: ' + str(int(mass)) + ')'
        CMPltName = newOutputDir + '/ConfusionMatrix.png'
        DrawCM(cnf_matrix, True, titleCM, CMPltName)

    ### Saving performance parameters
    logFile.write('\nTestLoss: ' + str(testLoss) + '\nTestAccuracy: ' + str(testAccuracy) + '\nROC_AUC: ' + str(roc_auc))

    ### Prediction on signal and background separately
    X_train_signal = X_train[y_train == 1]
    X_train_bkg = X_train[y_train != 1]
    #print('Running model prediction on X_train_signal_mass')
    m_train_signal = []
    for trainEvent in X_train_signal:
        m_train_signal.append(trainEvent[-1])
    X_train_signal_mass = X_train_signal[m_train_signal == mass]
    yhat_train_signal_mass = model.predict(X_train_signal_mass, batch_size = 2048)
    #print('Running model prediction on X_train_bkg')
    for bkg in X_train_bkg:
        bkg[0] = mass
    yhat_train_bkg_mass = model.predict(X_train_bkg, batch_size = 2048)
    #print('Running model prediction on X_test_signal_mass')
    yhat_test_signal_mass = model.predict(X_test_signal_mass, batch_size = 2048)
    #print('Running model prediction on X_test_bkg')
    yhat_test_bkg_mass = model.predict(X_test_bkg, batch_size = 2048)

    if plot:
        ### Plotting scores
        titleScores = 'Scores (mass hypotesis: ' + str(int(mass)) + ')'
        ScoresPltName = newOutputDir + '/Scores.png'
        DrawScores(yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, titleScores, ScoresPltName)

    ### Closing the logFile
    logFile.close()
    print('Saved ' + logFileName)

