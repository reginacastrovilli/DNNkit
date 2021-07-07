from Functions import *

savePlot = True
NN = 'DNN'
useWeights = False
cutTrainEvents = True
print(Fore.BLUE + '         useWeights = ' + str(useWeights))
print(Fore.BLUE + '     cutTrainEvents = ' + str(cutTrainEvents))

### Reading the command line
jetCollection, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass = ReadArgParser()

### Reading the configuration file
dfPath, InputFeatures, massColumnIndex = ReadConfig(analysis, jetCollection)
dfPath += analysis + '/' + channel + '/' + signal + '/'

### Loading input data
X_train, X_test, y_train, y_test, m_train_unscaled, m_test_unscaled, X_input = LoadData(dfPath, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts)

### Building the DNN
model = BuildDNN(X_input.shape[1] - 1 , numberOfNodes, numberOfLayers, dropout) # mass won't be given as input to the DNN

### Dividing signal from background
X_test_signal = X_test[y_test == 1]
X_test_bkg = X_test[y_test != 1]
m_test_unscaled_signal = m_test_unscaled[y_test == 1]
X_train_signal = X_train[y_train == 1]
X_train_bkg = X_train[y_train != 1]
m_train_unscaled_signal = m_train_unscaled[y_train == 1]

### Saving unscaled train signal masses
unscaledTrainMassPointsList = list(dict.fromkeys(list(m_train_unscaled_signal)))

### Extracting scaled test/train signal masses
m_test_signal = X_test_signal[:, massColumnIndex]
m_train_signal = X_train_signal[:, massColumnIndex]
scaledTrainMassPointsList = list(dict.fromkeys(list(m_train_signal)))

### Deleting mass 
X_test_signal = np.delete(X_test_signal, massColumnIndex, axis = 1)
X_train_signal = np.delete(X_train_signal, massColumnIndex, axis = 1) 
X_test_bkg = np.delete(X_test_bkg, massColumnIndex, axis = 1)
X_train_bkg = np.delete(X_train_bkg, massColumnIndex, axis = 1)

foundTestMass = False
for mass in scaledTrainMassPointsList:

    ### Associating the scaled mass to the unscaled one
    unscaledMass = unscaledTrainMassPointsList[scaledTrainMassPointsList.index(mass)]
    if unscaledMass != testMass:
        continue
    foundTestMass = True

    ### Creating the output directory
    outputDir = dfPath + background + '/' + NN + '/useWeights' + str(useWeights) + '/cutTrainEvents' + str(cutTrainEvents) + '/' + str(int(unscaledMass)) ### sposterei il bkg in dfPath e quindi nell'output di datapreprocessing
    print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
    
    ### Creating the logFile
    logFileName = outputDir + '/logFile.txt'
    logFile = open(logFileName, 'w')
    logString = WritingLogFile(dfPath, X_input, X_test, y_test, X_train, y_train, InputFeatures, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, useWeights)
    logFile.write(logString)

    ### Selecting signal events with the same mass
    X_train_signal_mass = X_train_signal[m_train_signal == mass]
    X_test_signal_mass = X_test_signal[m_test_signal == mass] 

    ### Creating new extended arrays by adding value 1 (0) to signal (background) events (this information is needed in order to shuffle data properly)
    X_train_signal_mass_ext = np.insert(X_train_signal_mass, X_train_signal_mass.shape[1], 1, axis = 1)
    X_train_bkg_ext = np.insert(X_train_bkg, X_train_bkg.shape[1], 0, axis = 1)
    X_test_signal_mass_ext = np.insert(X_test_signal_mass, X_test_signal_mass.shape[1], 1, axis = 1)
    X_test_bkg_ext = np.insert(X_test_bkg, X_test_bkg.shape[1], 0, axis = 1)

    ### Putting signal and background events back together
    X_train_mass_ext = np.concatenate((X_train_signal_mass_ext, X_train_bkg_ext), axis = 0)
    X_test_mass_ext = np.concatenate((X_test_signal_mass_ext, X_test_bkg_ext), axis = 0)

    logFile.write('\nNumber of test events with mass ' + str(unscaledMass) + ' GeV: ' + str(X_test_mass_ext.shape[0]) + ' (' + str(X_test_signal_mass.shape[0]) + ' signal)\nNumber of train events with mass ' + str(unscaledMass) + ' GeV: ' + str(X_train_mass_ext.shape[0]) + ' (' + str(X_train_signal_mass.shape[0]) + ' signal)')

    ### Shuffling data
    X_train_mass_ext = ShufflingData(X_train_mass_ext)
    X_test_mass_ext = ShufflingData(X_test_mass_ext)

    ### Extracting y_mass from X_mass_ext
    y_train_mass = X_train_mass_ext[:, X_train_mass_ext.shape[1] - 1]
    y_test_mass = X_test_mass_ext[:, X_test_mass_ext.shape[1] - 1]

    ### Deleting y_mass
    X_train_mass = np.delete(X_train_mass_ext, X_train_mass_ext.shape[1] - 1, axis = 1)
    X_test_mass = np.delete(X_test_mass_ext, X_test_mass_ext.shape[1] - 1, axis = 1)

    ### Creating the validation dataset
    stop  = int(validationFraction * X_train_mass.shape[0])
    X_validation_mass = X_train_mass[:stop]
    y_validation_mass = y_train_mass[:stop]
    X_train_mass = X_train_mass[stop:]
    y_train_mass = y_train_mass[stop:]

    if cutTrainEvents == True:
        X_train_mass, y_train_mass = cutEvents(X_train_mass, y_train_mass)

    logFile.write('\nNumber of real train events (without validation set): ' + str(X_train_mass.shape[0]) + ' (' + str(sum(y_train_mass)) + ' signal and ' + str(len(y_train_mass) - sum(y_train_mass)) + ' background)')

    for ivar in range (0, X_train_mass.shape[1]):
        plt.hist(X_train_mass[:, ivar], bins = 50, histtype = 'step', lw = 2, color = 'blue', density = False)
        plt.title('X_train_mass column #: ' + str(ivar))
        #plt.show()
        plt.savefig(outputDir + '/Column' + str(ivar) + '.png')
        plt.clf()

    ### Weighting train events
    w_train_mass = None
    if(useWeights == True):
        w_train_signal, w_train_bkg, w_train_mass = EventsWeight(y_train_mass)
        logFile.write('\nWeight for train signal events: ' + str(w_train_signal) + ', for background train events: ' + str(w_train_bkg))

    ### Training
    callbacks = [
        # If we don't have a decrease of the loss for 11 epochs, terminate training.
        EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')
    ]
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

    modelMetricsHistory = model.fit(X_train_mass, y_train_mass, sample_weight = w_train_mass, epochs = numberOfEpochs, batch_size = 2048, validation_data = (X_validation_mass, y_validation_mass), verbose = 1, callbacks = callbacks)

    ### Saving to files
    SaveModel(model, X_input, InputFeatures, outputDir)
    #model.save(outputDir + '/model/') ### per Martino

    ### Evaluating the performance of the DNN
    testLoss, testAccuracy = EvaluatePerformance(model, X_test_mass, y_test_mass)
    
    logFile.write('\nTest Loss: ' + str(testLoss) + '\nTest Accuracy: ' + str(testAccuracy))
    if savePlot:
        DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, useWeights, cutTrainEvents, unscaledMass)
        DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, useWeights, cutTrainEvents, unscaledMass)

    ### Prediction on the full test sample
    yhat_test = model.predict(X_test_mass, batch_size = 2048) 
    
    ### Evaluating ROC
    fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test)
    roc_auc = auc(fpr, tpr)
    print(format(Fore.BLUE + 'old ROC_AUC: ' + str(roc_auc)))
    logFile.write('\nold ROC_AUC: ' + str(roc_auc))

    if savePlot:
        DrawROC(fpr, tpr, roc_auc, outputDir, unscaledMass)

    ### Prediction on signal and background separately
    yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)

    ### Calculating area under ROC curve (AUC), background rejection and saving plots
    AUC, WP, WP_rej = DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, savePlot, useWeights, cutTrainEvents)
    print(Fore.BLUE + 'AUC: ' + str(AUC))
    logFile.write('\nAUC: ' + str(AUC) + '\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(WP_rej))
    if savePlot:
        DrawCM(yhat_test, y_test_mass, True, outputDir, unscaledMass)

    ### Closing the logFile
    logFile.close()
    print('Saved ' + logFileName)
