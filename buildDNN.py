from Functions import *

### Setting a seed for reproducibility
import tensorflow as tf
tf.random.set_seed(1234)

savePlot = True
NN = 'DNN'
batchSize = 2048

### Reading the command line
jetCollection, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass = ReadArgParser()

### Reading the configuration file
dfPath, InputFeatures = ReadConfig(analysis, jetCollection) 
dfPath += analysis + '/' + channel + '/' + str(signal) + '/' + background

### Loading input data
data_train, data_test, X_train_unscaled, m_train_unscaled, m_test_unscaled = LoadData(dfPath, jetCollection, str(signal), analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures) 

### Removing 'mass' from the list of variables that will be given as input to the DNN and from the unscaled train sample (otherwise files saved with "SaveModel" will have the wrong format) 
InputFeatures.remove('mass')
X_train_unscaled = X_train_unscaled[InputFeatures]

### Dividing signal from background
data_test_signal = data_test[data_test['isSignal'] == 1]
data_test_bkg = data_test[data_test['isSignal'] == 0]
m_test_unscaled_signal = m_test_unscaled[data_test['isSignal'] == 1]
data_train_signal = data_train[data_train['isSignal'] == 1]
data_train_bkg = data_train[data_train['isSignal'] == 0]
m_train_unscaled_signal = m_train_unscaled[data_train['isSignal'] == 1]

### Saving unscaled train signal masses
unscaledTrainMassPointsList = list(dict.fromkeys(list(m_train_unscaled_signal)))

### Extracting scaled test/train signal masses
#m_test_signal = data_test_signal['mass']
m_train_signal = data_train_signal['mass']
scaledTrainMassPointsList = list(dict.fromkeys(list(m_train_signal)))

if testMass == ['all']:
    testMass = [] ############################ serve?
    testMass = list(str(int(item)) for item in set(list(m_test_unscaled_signal)))

for unscaledMass in testMass:
    unscaledMass = int(unscaledMass)

    ### Checking whether there are train events with the selected mass
    if unscaledMass not in unscaledTrainMassPointsList:
        print(Fore.RED + 'No train signal with mass ' + str(unscaledMass))
        continue
    
    ### Associating the unscaled mass to the scaled one
    mass = scaledTrainMassPointsList[unscaledTrainMassPointsList.index(unscaledMass)]

    ### Creating the output directory
    outputDir = dfPath + '/' + NN + '/' + str(int(unscaledMass))
    print(format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

    ### Creating the logFile
    logFileName = outputDir + '/logFile.txt'
    logFile = open(logFileName, 'w')
    logString = WriteLogFile(numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, InputFeatures, dfPath)
    logFile.write(logString)
    logFile.write('\nNumber of train events: ' + str(len(data_train)) + ' (' + str(len(data_train_signal)) + ' signal and ' + str(len(data_train_bkg)) + ' background)' + '\nNumber of test events: ' + str(len(data_test)) + ' (' + str(len(data_test_signal)) + ' signal and ' + str(len(data_test_bkg)) + ' background)')

    ### Selecting signal events with the same mass
    data_train_signal_mass = data_train_signal[data_train_signal['mass'] == mass]
    data_test_signal_mass = data_test_signal[data_test_signal['mass'] == mass]
    logFile.write('\nNumber of train signal events with mass ' + str(unscaledMass) + ': ' + str(len(data_train_signal_mass)) + '\nNumber of test signal events with mass ' + str(unscaledMass) + ': ' + str(len(data_test_signal_mass)))

    ### Putting signal and background events back together
    data_train_mass = pd.concat([data_train_signal_mass, data_train_bkg], ignore_index = True)
    data_test_mass = pd.concat([data_test_signal_mass, data_test_bkg], ignore_index = True)

    ### Shuffling data
    data_train_mass = ShufflingData(data_train_mass)
    data_test_mass = ShufflingData(data_test_mass)

    ### Extracting y_mass and origin_mass as numpy arrays
    y_train_mass = np.asarray(data_train_mass['isSignal'].values).astype(np.float32)
    y_test_mass = np.asarray(data_test_mass['isSignal'].values).astype(np.float32)
    origin_train_mass = np.array(data_train_mass['origin'].values)
    origin_test_mass = np.array(data_test_mass['origin'].values)

    ### Selecting only the variables to give to the DNN
    X_train_mass = data_train_mass[InputFeatures]
    X_test_mass = data_test_mass[InputFeatures]

    ### Converting pandas dataframes into numpy arrays
    X_train_mass = np.asarray(X_train_mass.values).astype(np.float32)
    X_test_mass = np.asarray(X_test_mass.values).astype(np.float32)

    ### Weighting train events
    w_train_mass, origins_list, origins_numbers = weightEvents(origin_train_mass)
    logFile.write('\nOrigins list: ' + str(origins_list) + '\nNumber of events with the corresponding origin: ' + str(origins_numbers))
    #logFile.write('\nWeights for train events: ' + str(list(set(list(w_train_mass)))))

    ### Building the model, compiling and training
    model = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout)
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', weighted_metrics = ['accuracy'])
        
    print(Fore.BLUE + 'Training the DNN on train events with mass ' + str(int(unscaledMass)))
    modelMetricsHistory = model.fit(X_train_mass, y_train_mass, sample_weight = w_train_mass, epochs = numberOfEpochs, batch_size = batchSize, validation_split = validationFraction, verbose = True, callbacks = EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss', restore_best_weights = True))

    ### Saving to files
    SaveModel(model, X_train_unscaled, outputDir)

    ### Evaluating the performance of the DNN and writing results to the log file
    print(Fore.BLUE + 'Evaluating the performance of the DNN on test events with mass ' + str(int(unscaledMass)))
    testLoss, testAccuracy = EvaluatePerformance(model, X_test_mass, y_test_mass)
    logFile.write('\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy))

    ### Drawing accuracy and loss
    if savePlot:
        DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, unscaledMass)
        DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, unscaledMass)

    ### Prediction on the whole test sample and confusion matrix
    yhat_test = model.predict(X_test_mass, batch_size = batchSize) 
    if savePlot:
        DrawCM(yhat_test, y_test_mass, True, outputDir, unscaledMass, background)

    ### Prediction on signal and background separately 
    X_train_signal_mass = X_train_mass[y_train_mass == 1]
    X_test_signal_mass = X_test_mass[y_test_mass == 1]
    X_train_bkg = X_train_mass[y_train_mass == 0]
    X_test_bkg = X_test_mass[y_test_mass == 0]
    yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)
    '''
    scoresFile = open(outputDir + '/Scores_train_signal.txt', 'w')
    for score in yhat_train_signal:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()

    scoresFile = open(outputDir + '/Scores_train_bkg.txt', 'w')
    for score in yhat_train_bkg:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()

    scoresFile = open(outputDir + '/Scores_test_signal.txt', 'w')
    for score in yhat_test_signal:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()

    scoresFile = open(outputDir + '/Scores_test_bkg.txt', 'w')
    for score in yhat_test_bkg:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()
    '''
    ### Drawing scores, ROC and background rejection
    AUC, WP, WP_rej = DrawEfficiency(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, savePlot)
    print(Fore.BLUE + 'AUC (Area Under ROC Curve): ' + str(AUC))
    logFile.write('\nAUC: ' + str(AUC) + '\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(WP_rej))

    '''
    ### Dividing sample by origin
    originsBkgTest = list(background.split('_'))
    if len(originsBkgTest) > 1:
        for origin in originsBkgTest:
            print(Fore.BLUE + 'Evaluating the performance of the DNN on events with mass ' + str(unscaledMass) + ' and origin = \'' + signal + '\' or \'' + origin + '\'')
            logFile.write('\nOrigin: ' + origin)
            originsTest = origin_test_mass.copy()

            ### Selecting events with origin equal to signal or the background considered
            originsTest = np.where(originsTest == signal, 1, originsTest)
            originsTest = np.where(originsTest == origin, 1, originsTest)
            X_test_mass_origin = X_test_mass[originsTest == 1]
            y_test_mass_origin = y_test_mass[originsTest == 1]

            ### Prediction on the whole test sample and confusion matrix
            yhat_test_origin = model.predict(X_test_mass_origin, batch_size = batchSize) 
            if savePlot:
                DrawCM(yhat_test_origin, y_test_mass_origin, True, outputDir, unscaledMass, origin)

            ### Prediction on signal and background separately
            X_train_bkg_origin = X_train_mass[origin_train_mass == origin]
            X_test_bkg_origin = X_test_mass[origin_test_mass == origin]
            logFile.write('\nNumber of background train events with origin ' + origin + ': ' + str(len(X_train_bkg_origin)) + '\nNumber of background test events with origin ' + origin + ': ' + str(len(X_test_bkg_origin)))
            logFile.write('\nNumber of test events with origin ' + origin + ': ' + str(len(X_test_mass_origin)))
            yhat_train_signal_origin, yhat_train_bkg_origin, yhat_test_signal_origin, yhat_test_bkg_origin = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg_origin, X_test_signal_mass, X_test_bkg_origin) ### quelle sul segnale non sono cambiate!

            ### Drawing scores, ROC and background rejection
            AUC, WP, WP_rej = DrawEfficiency(yhat_train_signal_origin, yhat_test_signal_origin, yhat_train_bkg_origin, yhat_test_bkg_origin, outputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, origin, savePlot, useWeights, cutTrainEvents)
            print(Fore.BLUE + 'AUC (Area Under ROC Curve): ' + str(AUC))
            logFile.write('\nAUC: ' + str(AUC) + '\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(WP_rej))
    '''
    '''
            with open(outputDir + '/BkgRejection_' + origin + '_last.txt', 'a') as BkgRejectionFile:
                BkgRejectionFile.write(str(WP_rej[0]) + '\n')

            np.savetxt('data_train_signal_mass.csv', data_train_signal_mass, delimiter = ',', fmt = '%s')
            np.savetxt('data_train_bkg.csv', data_train_bkg, delimiter = ',', fmt = '%s')

            np.savetxt('data_test_signal_mass.csv', data_test_signal_mass, delimiter = ',', fmt = '%s')
            np.savetxt('data_test_bkg.csv', data_test_bkg, delimiter = ',', fmt = '%s')
    '''

    ### Closing the logFile
    logFile.close()
    print(Fore.GREEN + 'Saved ' + logFileName)
