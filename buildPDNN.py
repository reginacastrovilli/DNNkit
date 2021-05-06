### Make sure that 'DSID' is the last input variable
from Functions import *

plot = True
NN = 'pDNN'

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

### Splitting into test/train samples                                                                                                                        
Ntrain_stop = int(round(X_input.shape[0] * trainingFraction))
X_train = X_input[:Ntrain_stop]
X_test = X_input[Ntrain_stop:]
y_train = y_input[:Ntrain_stop]
y_test = y_input[Ntrain_stop:]

### Dividing signal from background
X_test_signal_unscaled = X_test[y_test == 1]
X_test_bkg_unscaled = X_test[y_test != 1]
X_train_signal_unscaled = X_train[y_train == 1]
X_train_bkg_unscaled = X_train[y_train != 1]

logString = '\nNumber of train events: ' + str(X_train.shape[0]) + ' (' + str(X_train_signal_unscaled.shape[0]) + ' signal and ' + str(X_train_bkg_unscaled.shape[0]) + ' background)' + '\nNumber of test events: ' + str(X_test.shape[0]) + ' (' + str(X_test_signal_unscaled.shape[0]) + ' signal and ' + str(X_test_bkg_unscaled.shape[0]) + ' background)'
logFile.write(logString) 
logInfo = logInfo + logString

### Saving unscaled test mass values, sorted in ascending order
unscaledMass = []
for event in X_test_signal_unscaled:
    unscaledMass.append(event[-1])
unscaledMassPointsList = sorted(list(set(unscaledMass)))
print('Mass points: ' + str(unscaledMassPointsList))
logString = '\nMass points: ' + str(unscaledMassPointsList)
logFile.write(logString)
logInfo = logInfo + logString

### Weighting train events
w_train_signal, w_train_bkg = EventsWeight(X_train_signal_unscaled, X_train_bkg_unscaled)
logString = '\nWeight for signal train events: ' + str(w_train_signal) + '\nWeight for background train events: ' + str(w_train_bkg)
logFile.write(logString)
logInfo += logString

### Creating new extended train arrays by adding W_train (this information is needed in order to shuffle data properly)
X_train_signal_ext = np.insert(X_train_signal_unscaled, X_train_signal_unscaled.shape[1], w_train_signal, axis = 1)
X_train_bkg_ext = np.insert(X_train_bkg_unscaled, X_train_bkg_unscaled.shape[1], w_train_bkg, axis = 1)

### Creating new extended train/test arrays by adding value 1 (0) to signal (background) events (this information is needed in order to shuffle data properly)    
X_train_signal_ext = np.insert(X_train_signal_ext, X_train_signal_ext.shape[1], 1, axis = 1)
X_train_bkg_ext = np.insert(X_train_bkg_ext, X_train_bkg_ext.shape[1], 0, axis = 1)
X_test_signal_ext = np.insert(X_test_signal_unscaled, X_test_signal_unscaled.shape[1], 1, axis = 1)
X_test_bkg_ext = np.insert(X_test_bkg_unscaled, X_test_bkg_unscaled.shape[1], 0, axis = 1)

### Putting train/test signal and background events back together
X_train_ext_unscaled = np.concatenate((X_train_signal_ext, X_train_bkg_ext), axis = 0)
X_test_ext_unscaled = np.concatenate((X_test_signal_ext, X_test_bkg_ext), axis = 0)

### Shuffling train/test data
X_train_ext_unscaled = ShufflingData(X_train_ext_unscaled)
X_test_ext_unscaled = ShufflingData(X_test_ext_unscaled)

### Extracting y_train/y_test from X_train_ext/X_test_ext
y_train = X_train_ext_unscaled[:, X_train_ext_unscaled.shape[1] - 1]
y_test = X_test_ext_unscaled[:, X_test_ext_unscaled.shape[1] - 1]

### Deleting y_train/y_test from X_train_ext/X_test_ext
X_train_ext_unscaled = np.delete(X_train_ext_unscaled, X_train_ext_unscaled.shape[1] - 1, axis = 1)
X_test_unscaled = np.delete(X_test_ext_unscaled, X_test_ext_unscaled.shape[1] - 1, axis = 1)

### Extracting w_train from X_train_ext
w_train = X_train_ext_unscaled[:, X_train_ext_unscaled.shape[1] - 1]

### Deleting w_train from X_train_ext
X_train_unscaled = np.delete(X_train_ext_unscaled, X_train_ext_unscaled.shape[1] - 1, axis = 1)

### Scaling train/test data
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.fit_transform(X_test_unscaled)

### Extracting scaled signal and background events
X_test_signal = X_test[y_test == 1]
X_test_bkg = X_test[y_test != 1]
X_train_signal = X_train[y_train == 1]
X_train_bkg = X_train[y_train != 1]

### Extracting scaled signal train/test masses
m_test_signal = X_test_signal[:, X_test_signal.shape[1] - 1]
m_train_signal = X_train_signal[:, X_train_signal.shape[1] - 1]

### Saving scaled signal train/test massed, sorted in ascending order (the same unscaled train/test mass values are scaled into different train/test mass values. If we sort them in ascending order we should keep the correspondence between scaled and unscaled values)
scaledTestMassPointsList = sorted(list(set(m_test_signal)))
scaledTrainMassPointsList = sorted(list(set(m_train_signal)))

### Building the pDNN
n_dim = X_train.shape[1]

model = BuildDNN(n_dim, numberOfNodes, numberOfLayers, dropout)
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### Training
callbacks = [
    # If we don't have a decrease of the loss for 11 epochs, terminate training.
    EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')
]

modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, callbacks = callbacks)

### Evaluating the performance of the pDNN
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

### Saving the model
SaveModel(model, outputDir)

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
    X_train_signal_mass = X_train_signal[m_train_signal == scaledTrainMassPointsList[counter]]

    ### Assigning the same mass value to train background events
    for bkg in X_train_bkg:
          bkg[X_train_bkg.shape[1] - 1] = scaledTrainMassPointsList[counter]
          
    ### Prediction
    yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)

    if plot:
        ### Plotting scores
        DrawScores(yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, newOutputDir, NN, unscaledMass)

    ### Closing the newLogFile
    newLogFile.close()
    print('Saved ' + newLogFileName)
    counter = counter + 1
