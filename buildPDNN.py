### Make sure that 'DSID' is the last input variable
from Functions import *
from keras.models import load_model
import json 
from json import JSONEncoder

plot = True
NN = 'PDNN'

### Reading the command line
analysis, channel, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, trainingFraction = ReadArgParser()

### Reading the configuration file
dfPath, modelPath, InputFeatures = ReadConfig(analysis)

### Creating the output directory
outputDir = modelPath + NN + '/' + analysis + '/' + channel
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

### Creating the logFile                                                                                                                                         
logFileName = outputDir + '/logFile.txt'
logFile = open(logFileName, 'w')

### Loading data and creating input arrays
X_input, y_input, dfInput = LoadData(dfPath, analysis, channel, InputFeatures)

logInfo = ''
logString = 'Input dataframe: ' + dfInput + '\nNumber of input events: ' + str(X_input.shape[0]) + '\nInputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nTraining fraction: ' + str(trainingFraction)
logFile.write(logString)
logInfo += logString

### Saving unscaled test mass values, sorted in ascending order
unscaledMass = []
for event in X_input:
    unscaledMass.append(event[-1])
unscaledMassPointsList = sorted(list(set(unscaledMass)))
print('Mass points: ' + str(unscaledMassPointsList))
logString = '\nMass points: ' + str(unscaledMassPointsList)
logFile.write(logString)
logInfo = logInfo + logString

### Scaling data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_input)

### Splitting into test/train samples                                                                                                                        
Ntrain_stop = int(round(X_scaled.shape[0] * trainingFraction))
X_train = X_scaled[:Ntrain_stop]
X_test = X_scaled[Ntrain_stop:]
y_train = y_input[:Ntrain_stop]
y_test = y_input[Ntrain_stop:]

### Dividing signal from background
X_test_signal = X_test[y_test == 1]
X_test_bkg = X_test[y_test != 1]
X_train_signal = X_train[y_train == 1]
X_train_bkg = X_train[y_train != 1]

logString = '\nNumber of train events: ' + str(X_train.shape[0]) + ' (' + str(X_train_signal.shape[0]) + ' signal and ' + str(X_train_bkg.shape[0]) + ' background)' + '\nNumber of test events: ' + str(X_test.shape[0]) + ' (' + str(X_test_signal.shape[0]) + ' signal and ' + str(X_test_bkg.shape[0]) + ' background)'
logFile.write(logString) 
logInfo = logInfo + logString

### Extracting scaled signal test masses
m_test_signal = X_test_signal[:, X_test_signal.shape[1] - 1]

### Saving scaled signal train/test massed, sorted in ascending order (the same unscaled train/test mass values are scaled into different train/test mass values. If we sort them in ascending order we should keep the correspondence between scaled and unscaled values)
scaledTestMassPointsList = sorted(list(set(m_test_signal)))

### Weighting train events
w_train_signal, w_train_bkg = EventsWeight(X_train_signal, X_train_bkg)
logString = '\nWeight for signal train events: ' + str(w_train_signal) + '\nWeight for background train events: ' + str(w_train_bkg)
logFile.write(logString)
logInfo += logString

y_train = []
w_train = []
for event in range(len(X_train)):
    if event < len(X_train_signal):
        y_train.append(1)
        w_train.append(w_train_signal)
    else:
        y_train.append(0)
        w_train.append(w_train_bkg)
y_train = np.array(y_train)
w_train = np.array(w_train)

X_train = np.concatenate((X_train_signal, X_train_bkg), axis = 0)

### Building the PDNN
n_dim = X_train.shape[1]

model = BuildDNN(n_dim, numberOfNodes, numberOfLayers, dropout)
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### Training
callbacks = [
    # If we don't have a decrease of the loss for 11 epochs, terminate training.
    EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')#, ModelCheckpoint('model.hdf5', save_weights_only = False, monitor = 'val_loss', mode = 'min', verbose = True, save_best_only = True)
]

modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, shuffle = True, callbacks = callbacks)

### Saving to files
SaveModel(model, X_input, InputFeatures, outputDir)

### Evaluating the performance of the PDNN
testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test)

logString = '\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy)
logFile.write(logString)
logInfo += logString

if plot:
    ### Drawing training history
    DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN)
    DrawLoss(modelMetricsHistory, testLoss, outputDir, NN)

logFile.close()
print('Saved ' + logFileName)

### Prediction on the full test sample
counter = 0

for mass in scaledTestMassPointsList:
    
    ### Associating the scaled mass to the unscaled one
    unscaledMass = unscaledMassPointsList[counter]
    
    ### Creating new output directory and log file
    newOutputDir = outputDir + '/' + str(int(unscaledMass))
    print (format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
    newLogFileName = newOutputDir + '/logFile.txt'
    newLogFile = open(newLogFileName, 'w')
    newLogFile.write(logInfo + '\nMass point analyzed: ' + str(unscaledMass))

    ### Selecting only test signal events with the same mass value
    X_test_signal_mass = X_test_signal[m_test_signal == mass]
    newLogFile.write('\nNumber of test events with this mass: ' + str(X_test_signal_mass.shape[0])) 

    ### Assigning the same mass value to test background events
    for bkg in X_test_bkg:
        bkg[X_test_bkg.shape[1] - 1] = mass

    ### Creating a new extended test array by adding value 1 (0) to signal (background) events (this information is needed in order to shuffle data properly)
    X_test_signal_mass_ext = np.insert(X_test_signal_mass, X_test_signal_mass.shape[1], 1, axis = 1)
    X_test_bkg_ext = np.insert(X_test_bkg, X_test_bkg.shape[1], 0, axis = 1)

    ### Putting signal and background events back togheter
    X_test_mass_ext = np.concatenate((X_test_signal_mass_ext, X_test_bkg_ext), axis = 0)

    ### Shuffling data
    X_test_mass_ext = ShufflingData(X_test_mass_ext)

    ### Extracting y_mass from X_mass_ext
    y_test_mass = X_test_mass_ext[:, X_test_mass_ext.shape[1] - 1]

    ### Deleting y_mass from X_mass
    X_test_mass = np.delete(X_test_mass_ext, X_test_mass_ext.shape[1] - 1, axis = 1) 

    ### Prediction
    yhat_test_mass = model.predict(X_test_mass, batch_size = 2048)

    ### Evaluating ROC
    fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)
    roc_auc = auc(fpr, tpr)
    print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))

    if plot:
        ### Plotting ROC
        DrawROC(fpr, tpr, roc_auc, newOutputDir, unscaledMass)

        ### Plotting confusion matrix
        DrawCM(yhat_test_mass, y_test_mass, True, newOutputDir, unscaledMass)

    newLogFile.write('\nROC_AUC: ' + str(roc_auc))

    ###### Prediction on signal and background separately
    ### Selecting train signal events with the same mass
    m_train_signal = []
    for trainEvent in X_train_signal:
        m_train_signal.append(trainEvent[-1])
    X_train_signal_mass = X_train_signal[m_train_signal == scaledTestMassPointsList[counter]]

    ### Assigning the same mass value to train background events
    for bkg in X_train_bkg:
          bkg[X_train_bkg.shape[1] - 1] = scaledTestMassPointsList[counter]
          
    ### Prediction
    yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)

    if plot:
        ### Plotting scores
        DrawScores(yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, newOutputDir, NN, unscaledMass)

    ### Closing the newLogFile
    newLogFile.close()
    print('Saved ' + newLogFileName)
    counter = counter + 1
