from Functions import *

### Setting a seed for reproducibility
tf.random.set_seed(1234)

savePlot = True
NN = 'PDNN'
batchSize = 2048

### Reading the command line
jetCollection, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass = ReadArgParser()
originsBkgTest = list(background.split('_'))

### Reading the configuration file
dfPath, InputFeatures = ReadConfig(analysis, jetCollection)
dfPath += analysis + '/' + channel + '/' + signal + '/' + background + '/'

### Creating the output directory and the logFile
outputDir = dfPath + NN
print(format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
logFileName = outputDir + '/logFile.txt'
logFile = open(logFileName, 'w')
logInfo = ''
logString = WriteLogFile(numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, InputFeatures, dfPath)
logFile.write(logString)
logInfo += logString

### Loading input data
data_train, data_test, X_train_unscaled, m_train_unscaled, m_test_unscaled = LoadData(dfPath, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures)

### Extracting X and y arrays 
X_train = np.array(data_train[InputFeatures].values).astype(np.float32)
y_train = np.array(data_train['isSignal'].values).astype(np.float32)
X_test = np.array(data_test[InputFeatures].values).astype(np.float32)
y_test = np.array(data_test['isSignal'].values).astype(np.float32)

### Writing dataframes composition to the log file
logString = '\nNumber of train events: ' + str(len(X_train)) + ' (' + str(int(sum(y_train))) + ' signal and ' + str(int(len(y_train) - sum(y_train))) + ' background)' + '\nNumber of test events: ' + str(len(X_test)) + ' (' + str(int(sum(y_test))) + ' signal and ' + str(int(len(y_test) - sum(y_test))) + ' background)'
logFile.write(logString)
logInfo += logString

### Weighting train events
origin_train = np.array(data_train['origin'].values)
w_train, origins_list, origins_numbers, weights = weightEvents(origin_train)
logString = '\nOrigins list: ' + str(origins_list) + '\nNumber of events with the corresponding origin: ' + str(origins_numbers) + '\nWeights for each origin: ' + str(weights)
logFile.write(logString)
logInfo += logString

### Building and compiling the PDNN
model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout) 
model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
logString = '\nLoss: ' + Loss + '\nLearning rate: ' + str(learningRate) + '\nOptimizer: ' + str(Optimizer) + '\nweighted_metrics: ' + str(Metrics)
logFile.write(logString)
logInfo += logString

### Training
print(Fore.BLUE + 'Training the ' + NN)
modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = batchSize,  validation_split = validationFraction, verbose = 1, callbacks = EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss', restore_best_weights = True))

### Saving to files
SaveModel(model, X_train_unscaled, outputDir)

### Evaluating the performance of the PDNN on the test sample and writing results to the log file
print(Fore.BLUE + 'Evaluating the performance of the ' + NN)
testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test, batchSize)
logString = '\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy)
logFile.write(logString)
logInfo += logString

### Drawing accuracy and loss
if savePlot:
    DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background)
    DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background)

logFile.close()
print(Fore.GREEN + 'Saved ' + logFileName)

### Dividing signal from background
data_test_signal = data_test[y_test == 1]
data_test_bkg = data_test[y_test != 1]
X_train_signal = X_train[y_train == 1]
X_train_bkg = X_train[y_train != 1]

### Saving unscaled test signal mass values
m_test_unscaled_signal = m_test_unscaled[y_test == 1]
unscaledTestMassPointsList = list(dict.fromkeys(list(m_test_unscaled_signal)))

### Saving scaled test signal mass values
m_test_signal = data_test_signal['mass']
scaledTestMassPointsList = list(dict.fromkeys(list(m_test_signal)))

### If testMass = 'all', defining testMass as the list of test signal masses 
if testMass == ['all']:
    testMass = []
    testMass = list(str(int(item)) for item in set(list(m_test_unscaled_signal)))

for unscaledMass in testMass:
    unscaledMass = int(unscaledMass)

    ### Checking whether there are train events with the selected mass
    if unscaledMass not in unscaledTestMassPointsList:
        print(Fore.RED + 'No test signal with mass ' + str(unscaledMass))
        continue

    ### Associating the unscaled mass to the scaled one
    mass = scaledTestMassPointsList[unscaledTestMassPointsList.index(unscaledMass)]
    #print('Scaled mass: ' + str(mass))

    ### Creating new output directory and log file
    newOutputDir = outputDir + '/' + str(int(unscaledMass))
    print(format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
    newLogFileName = newOutputDir + '/logFile.txt'
    newLogFile = open(newLogFileName, 'w')

    ### Selecting only test signal events with the same mass value and saving them as an array
    data_test_signal_mass = data_test_signal[m_test_signal == mass]
    X_test_signal_mass = np.asarray(data_test_signal_mass[InputFeatures].values).astype(np.float32)
    newLogFile.write(logInfo + '\nNumber of test signal events with mass ' + str(int(unscaledMass)) + ' GeV: ' + str(len(X_test_signal_mass)))

    ### Assigning the same mass value to test background events and saving them as an array
    data_test_bkg = data_test_bkg.assign(mass = np.full(len(data_test_bkg), mass))
    X_test_bkg = np.asarray(data_test_bkg[InputFeatures].values).astype(np.float32)

    ### Selecting train signal events with the same mass
    m_train_signal = X_train_signal[:, InputFeatures.index('mass')]
    X_train_signal_mass = X_train_signal[m_train_signal == mass]

    ### Assigning the same mass value to train background events
    X_train_bkg[:, InputFeatures.index('mass')] = np.full(len(X_train_bkg), mass)

    ### Prediction on signal and background
    yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg, batchSize)

    scoresFile = open(newOutputDir + '/Scores_train_signal.txt', 'w')
    for score in yhat_train_signal_mass:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()

    scoresFile = open(newOutputDir + '/Scores_train_bkg.txt', 'w')
    for score in yhat_train_bkg_mass:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()                                                                                                                                               

    scoresFile = open(newOutputDir + '/Scores_test_signal.txt', 'w')
    for score in yhat_test_signal_mass:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()

    scoresFile = open(newOutputDir + '/Scores_test_bkg.txt', 'w')
    for score in yhat_test_bkg_mass:
        scoresFile.write(str(score) + '\n')
    scoresFile.close()

    ### Drawing confusion matrix
    yhat_test_mass = np.concatenate((yhat_test_signal_mass, yhat_test_bkg_mass))
    y_test_mass = np.concatenate((np.ones(len(yhat_test_signal_mass)), np.zeros(len(yhat_test_bkg_mass))))
    CMvalues = DrawCM(yhat_test_mass, y_test_mass, True, newOutputDir, unscaledMass, background, savePlot)
    newLogFile.write('\nTrue bkg, predicted bkg: ' + str(CMvalues[0]) + '\nTrue bkg, predicted signal: ' + str(CMvalues[1]) + '\nTrue signal, predicted bkg: ' + str(CMvalues[2]) + '\nTrue signal, predicted signal: ' + str(CMvalues[3]))

    ### Calculating area under ROC curve (AUC), background rejection and saving plots 
    AUC, WP, WP_rej = DrawEfficiency(yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, savePlot)
    print(Fore.BLUE + 'AUC: ' + str(AUC))
    newLogFile.write('\nAUC: ' + str(AUC) + '\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(WP_rej))

    ### Closing the newLogFile
    newLogFile.close()
    print(Fore.GREEN + 'Saved ' + newLogFileName)
