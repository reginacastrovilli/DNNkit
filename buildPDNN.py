from Functions import *

plot = True
NN = 'PDNN'
useWeights = False
print(Fore.BLUE + '         useWeights = ' + str(useWeights))

### Reading the command line
analysis, channel, signal, jetCollection, background, trainingFraction, preselectionCuts, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout = ReadArgParser()

### Reading the configuration file
#dfPath, modelPath, InputFeatures, massColumnIndex = ReadConfig(analysis, jetCollection)
dfPath, InputFeatures, massColumnIndex = ReadConfig(analysis, jetCollection)
dfPath += analysis + '/' + channel + '/' + signal + '/'

### Creating the output directory and the logFile
#outputDir = modelPath + signal + '/' + analysis + '/' + channel + '/' + NN + '/useWeights' + str(useWeights) ######modificare
outputDir = dfPath + NN + '/useWeights' + str(useWeights) ######modificare
print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
logFileName = outputDir + '/logFile.txt'
logFile = open(logFileName, 'w')

### Loading input data
X_train, X_test, y_train, y_test, m_train_unscaled, m_test_unscaled, X_input = LoadData(dfPath, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts)

logInfo = ''
logString = WritingLogFile(dfPath, X_input, X_test, y_test, X_train, y_train, InputFeatures, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, useWeights)
logFile.write(logString)
logInfo += logString

### Creating the validation dataset
stop  = int(validationFraction * X_train.shape[0])
X_validation = X_train[:stop]
y_validation = y_train[:stop]
X_train = X_train[stop:]
y_train = y_train[stop:]

logString = '\nNumber of real train events (without validation set): ' + str(X_train.shape[0]) + ' (' + str(sum(y_train)) + ' signal and ' + str(len(y_train) - sum(y_train)) + ' background)'
logFile.write(logString)
logInfo += logString

### Weighting train events
w_train = None
if(useWeights == True): 
    w_train_signal, w_train_bkg, w_train = EventsWeight(y_train)
    logString = '\nWeight for train signal events: ' + str(w_train_signal) + ', for background train events: ' + str(w_train_bkg)
    logFile.write(logString)
    logInfo += logString

### Building the PDNN
model = BuildDNN(X_train.shape[1], numberOfNodes, numberOfLayers, dropout)
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### Training
callbacks = [
    # If we don't have a decrease of the loss for 11 epochs, terminate training.
    EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')#, ModelCheckpoint('model.hdf5', save_weights_only = False, monitor = 'val_loss', mode = 'min', verbose = True, save_best_only = True)
]

#modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, shuffle = True, callbacks = callbacks)
modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = 2048,  validation_data = (X_validation, y_validation), verbose = 1, shuffle = True, callbacks = callbacks)

### Saving to files
SaveModel(model, X_input, InputFeatures, outputDir)
#model.save(outputDir + '/model/') ###per Martino

### Evaluating the performance of the PDNN
testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test)
logString = '\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy)
logFile.write(logString)
logInfo += logString

### Drawing training history
if plot:
    DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN)
    DrawLoss(modelMetricsHistory, testLoss, outputDir, NN)

#logFile.close()
print('Saved ' + logFileName)

###### Prediction on the full test sample

### Dividing signal from background
X_test_signal = X_test[y_test == 1]
X_test_bkg = X_test[y_test != 1]
m_test_unscaled_signal = m_test_unscaled[y_test == 1]
X_train_signal = X_train[y_train == 1]
X_train_bkg = X_train[y_train != 1]

### Saving unscaled test signal mass values
unscaledTestMassPointsList = list(dict.fromkeys(list(m_test_unscaled_signal)))

### Saving scaled signal test masses
m_test_signal = X_test_signal[:, massColumnIndex]
scaledTestMassPointsList = list(dict.fromkeys(list(m_test_signal)))

for mass in scaledTestMassPointsList:

    ### Associating the scaled mass to the unscaled one
    unscaledMass = unscaledTestMassPointsList[scaledTestMassPointsList.index(mass)]
    
    ### Creating new output directory and log file
    newOutputDir = outputDir + '/' + str(int(unscaledMass))
    print (format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
    newLogFileName = newOutputDir + '/logFile.txt'
    newLogFile = open(newLogFileName, 'w')

    ### Selecting only test signal events with the same mass value
    X_test_signal_mass = X_test_signal[m_test_signal == mass]
    newLogFile.write(logInfo + '\nNumber of test signal events with mass ' + str(unscaledMass) + ' GeV: ' + str(X_test_signal_mass.shape[0])) 

    ### Assigning the same mass value to test background events
    for bkg in X_test_bkg:
        bkg[massColumnIndex] = mass

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
    '''
    ### Evaluating ROC
    fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)
    roc_auc = auc(fpr, tpr)
    print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))
    newLogFile.write('\nROC_AUC: ' + str(roc_auc))

    ### Plotting ROC and confusion matrix
    if plot:
        DrawROC(fpr, tpr, roc_auc, newOutputDir, unscaledMass)
        DrawCM(yhat_test_mass, y_test_mass, True, newOutputDir, unscaledMass)
    '''
    ###### Prediction on signal and background separately
    ### Selecting train signal events with the same mass
    m_train_signal = []
    for trainEvent in X_train_signal:
        m_train_signal.append(trainEvent[-1])
    X_train_signal_mass = X_train_signal[m_train_signal == mass]
    newLogFile.write('\nNumber of train signal events with mass ' + str(unscaledMass) + ' GeV: ' + str(X_train_signal_mass.shape[0]))     

    ### Assigning the same mass value to train background events
    for bkg in X_train_bkg:
          bkg[massColumnIndex] = mass
          
    ### Prediction
    yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)

    ### Plotting scores
    if plot:
       bkg_eff, signal_eff = DrawScores(yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, newOutputDir, NN, unscaledMass)
       DrawROC(bkg_eff, signal_eff, newOutputDir, unscaledMass)
       DrawEfficiency(bkg_eff, signal_eff, newOutputDir, unscaledMass)
       DrawCM(yhat_test_mass, y_test_mass, True, newOutputDir, unscaledMass)

    ### Closing the newLogFile
    newLogFile.close()
    print('Saved ' + newLogFileName)
