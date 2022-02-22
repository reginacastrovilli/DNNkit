from Functions import *
#import tensorflow as tf
#import time
#start = time.time()
### Setting a seed for reproducibility
#tf.random.set_seed(1234)

#blockPrint()

NN = 'PDNN'
batchSize = 2048

### Reading the command line
tag, jetCollection, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass, doTrain, doTest, loop = ReadArgParser()

originsBkgTest = list(background.split('_'))

### Reading the configuration file
ntuplePath, dfPath, InputFeatures = ReadConfig(tag, analysis, jetCollection)
dfPath += analysis + '/' + channel + '/' + signal + '/' + background + '_fullStat/'#'/'
print(Fore.GREEN + 'Input files directory: ' + dfPath)
outputFileCommonName = jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background + '_' + NN

### Creating the output directory and the logFile
outputDir = dfPath + NN# + '_halfStat'
print(format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

logFileName = outputDir + '/logFile_' + outputFileCommonName + '.txt'
logFile = open(logFileName, 'w')
logInfo = ''
logString = WriteLogFile(tag, ntuplePath, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, InputFeatures, dfPath)
logFile.write(logString)
logInfo += logString

### Loading input data
#data_train, data_test, m_train_unscaled, m_test_unscaled, w_train = LoadData(dfPath, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures)
###X_train, X_test, y_train, y_test, m_train_unscaled, m_test_unscaled, w_train = LoadData(dfPath, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures)
data_train, data_test, X_train, X_test, y_train, y_test, w_train, w_test = LoadDataNew(dfPath, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures)
'''
### Se non peso piu' gli eventi qui potrei anche farmi restituire le y dal loading e i dati con solo le inputvariables
### Extracting X and y arrays 
X_train = np.array(data_train[InputFeatures].values).astype(np.float32)
y_train = np.array(data_train['isSignal'].values).astype(np.float32)
X_test = np.array(data_test[InputFeatures].values).astype(np.float32)
y_test = np.array(data_test['isSignal'].values).astype(np.float32)
'''
'''
### Writing dataframes composition to the log file
logString = '\nNumber of train events: ' + str(len(X_train)) + ' (' + str(int(sum(y_train))) + ' signal and ' + str(int(len(y_train) - sum(y_train))) + ' background)' + '\nNumber of test events: ' + str(len(X_test)) + ' (' + str(int(sum(y_test))) + ' signal and ' + str(int(len(y_test) - sum(y_test))) + ' background)'
bkgNumber = int(len(y_test) - sum(y_test))
logFile.write(logString)
logInfo += logString
'''
bkgRejFile = open(outputDir + '/BkgRejectionVsMass.txt', 'w')
bkgRejFile.write('Background rejection obtained using the version of the software 02-feb-2022, lepton masses as input feature, WP = 0.90, 0.94, 0.97, 0.99\n')
bkgRej90File = open(outputDir + '/BkgRejectionVsMassWP90.txt', 'w')
bkgRej94File = open(outputDir + '/BkgRejectionVsMassWP94.txt', 'w')
bkgRej97File = open(outputDir + '/BkgRejectionVsMassWP97.txt', 'w')
bkgRej99File = open(outputDir + '/BkgRejectionVsMassWP99.txt', 'w')

bkgRej90 = []
bkgRej94 = []
bkgRej97 = []
bkgRej99 = []
'''
### hp optimization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from keras_tuner.tuners import RandomSearch

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Dense(units = hp.Int('units', min_value = 32, max_value = 512, step = 128), input_dim = len(InputFeatures), activation = 'relu'))  
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step = 0.1)))
    for iLayer in range(hp.Int('layers', 2, 6)):
        model.add(tf.keras.layers.Dense(units = hp.Int('units_' + str(iLayer), 50, 100, step = 10), activation = 'relu'))                                   
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(iLayer), 0, 0.5, step = 0.1)))
    hp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]),
        decay_steps = 10000,
        decay_rate = 0.95)
    optimizer = hp.Choice('optimizer', values = ['RMSprop', 'Adam'])
    if optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = hp_lr)
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = hp_lr)
    else:
        raise
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', weighted_metrics = ['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective = 'val_accuracy',
    max_trials = 5, 
    executions_per_trial = 1,#2,
    directory = './tunerTrials/',  
    overwrite = True
)

print(tuner.search_space_summary())

tuner.search(X_train, y_train, sample_weight = w_train, epochs=5, validation_split = 0.2)
tuner.results_summary()
model = tuner.get_best_models()[0]
model.build(X_train.shape)

print(model.summary())

print('Number of nodes in layer 0:', tuner.get_best_hyperparameters()[0].get('units'))
print('Dropout in layer 0:', tuner.get_best_hyperparameters()[0].get('dropout'))
layersNumber = tuner.get_best_hyperparameters()[0].get('layers')
print('Number of hidden layers:', layersNumber)
for iLayer in range(1, layersNumber + 1):
    hp_nodes = 'units_' + str(iLayer - 1)
    hp_dropout = 'dropout_' + str(iLayer - 1)
    print('Number of nodes in hidden layer ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_nodes)))
    print('Dropout in hidden layer ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_dropout)))
print('Learning rate:', tuner.get_best_hyperparameters()[0].get('learning_rate'))
print('Optimizer:', tuner.get_best_hyperparameters()[0].get('optimizer'))
exit()
'''

bkgRej90Dict = {}
bkgRej94Dict = {}
bkgRej97Dict = {}
bkgRej99Dict = {}

for iLoop in range(loop):

    enablePrint()
    print(Fore.RED + 'loop: ' + str(iLoop))
    #blockPrint()

    ### Building and compiling the PDNN
    model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout) 
    model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
    if iLoop == 0:
        logString = '\nLoss: ' + Loss + '\nLearning rate: ' + str(learningRate) + '\nOptimizer: ' + str(Optimizer) + '\nweighted_metrics: ' + str(Metrics)
        logFile.write(logString)
        logInfo += logString
    
    if doTrain == False:
        from keras.models import model_from_json
        print(Fore.BLUE + 'Loading architecture and weights')
        architectureFile = open(outputDir + '/architecture.json', 'r')
        loadedModel = architectureFile.read()
        architectureFile.close()
        print(Fore.GREEN + 'Loaded ' + outputDir + '/achitecture.json') 
        model = model_from_json(loadedModel)
        model.load_weights(outputDir + '/weights.h5')
        print(Fore.GREEN + 'Loaded ' + outputDir + '/weights.h5')
        model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics) ### DEVE STARE QUI??? PRIMA NON LO CHIEDEVA

    if doTrain == True:
        ### Training
        patienceValue = 5
        print(Fore.BLUE + 'Training the ' + NN)
        modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = batchSize, validation_split = validationFraction, verbose = 1, shuffle = False, callbacks = EarlyStopping(verbose = True, patience = patienceValue, monitor = 'val_loss', restore_best_weights = True))

        ### Saving to files
        ##SaveModel(model, X_train_unscaled, outputDir)
        SaveModel(model, outputDir, NN)
        
    if doTest == True:
        ### Evaluating the performance of the PDNN on the test sample and writing results to the log file
        print(Fore.BLUE + 'Evaluating the performance of the ' + NN)
        testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test, w_test, batchSize)
        if iLoop == 0:
            logString = '\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy)
            logFile.write(logString)
            logInfo += logString
            
    else:
        testLoss = testAccuracy = None

    if doTrain and doTest:
        ### Drawing accuracy and loss
        DrawLoss(modelMetricsHistory, testLoss, patienceValue, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)
        DrawAccuracy(modelMetricsHistory, testAccuracy, patienceValue, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)
    #exit()
    if iLoop == 0:
        logFile.close()
        print(Fore.GREEN + 'Saved ' + logFileName)

    if doTest == False:
        exit()

    ### Dividing signal from background
    '''
    data_test_signal = data_test[y_test == 1]
    data_test_bkg = data_test[y_test != 1]
    X_train_signal = X_train[y_train == 1]
    X_train_bkg = X_train[y_train != 1]
    '''
    data_test_signal = data_test[data_test['isSignal'] == 1]
    data_test_bkg = data_test[data_test['isSignal'] != 1]
    data_train_signal = data_train[data_train['isSignal'] == 1]
    data_train_bkg = data_train[data_train['isSignal'] != 1]

    ### Saving unscaled test signal mass values
    #m_test_unscaled_signal = m_test_unscaled[y_test == 1]
    m_test_unscaled_signal = data_test_signal['unscaledMass']
    unscaledTestMassPointsList = list(dict.fromkeys(list(m_test_unscaled_signal)))
    '''
    ### Saving scaled test signal mass values
    m_test_signal = data_test_signal['mass']
    scaledTestMassPointsList = list(dict.fromkeys(list(m_test_signal)))
    '''    
    ### If testMass = 'all', defining testMass as the list of test signal masses 
    if testMass == ['all']:
        testMass = list(int(item) for item in set(list(m_test_unscaled_signal)))
    else:
        testMass = list(int(item) for item in testMass)
    testMass.sort()

    for unscaledMass in testMass:

        ### Checking whether there are test events with the selected mass
        if unscaledMass not in unscaledTestMassPointsList:
            print(Fore.RED + 'No test signal with mass ' + str(unscaledMass))
            continue
        '''
        ### Associating the unscaled mass to the scaled one
        mass = scaledTestMassPointsList[unscaledTestMassPointsList.index(unscaledMass)]
        '''
        ### Creating new output directory and log file
        newOutputDir = outputDir + '/' + str(int(unscaledMass))
        print(format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
        newLogFileName = newOutputDir + '/logFile_' + outputFileCommonName + '_' + str(unscaledMass) + '.txt'
        newLogFile = open(newLogFileName, 'w')

        if (iLoop == 0 and loop > 1):
            bkgRej90Dict[unscaledMass] = []
            bkgRej94Dict[unscaledMass] = []
            bkgRej97Dict[unscaledMass] = []
            bkgRej99Dict[unscaledMass] = []

        ### Selecting only test signal events with the same mass value and saving them as an array
        #data_test_signal_mass = data_test_signal[m_test_signal == mass]
        data_test_signal_mass = data_test_signal[data_test_signal['unscaledMass'] == unscaledMass]
        scaledMass = list(set(list(data_test_signal_mass['mass'])))[0]
        ### perche' non prendo direttamente le x?

        X_test_signal_mass = np.asarray(data_test_signal_mass[InputFeatures].values).astype(np.float32)
        newLogFile.write(logInfo + '\nNumber of test signal events with mass ' + str(int(unscaledMass)) + ' GeV: ' + str(len(X_test_signal_mass)))
        wMC_test_signal_mass = np.array(data_test_signal_mass['weight'])

        ### Assigning the same mass value to test background events and saving them as an array
        ### Fare direttamente su x ??
        data_test_bkg = data_test_bkg.assign(mass = np.full(len(data_test_bkg), scaledMass))
        X_test_bkg = np.asarray(data_test_bkg[InputFeatures].values).astype(np.float32)
        #wMC_test_bkg = np.asarray(data_test_bkg['weight'].values).astype(np.float32)
        wMC_test_bkg = np.array(data_test_bkg['weight'])
        #print(wMC_test_signal_mass)
        #print(wMC_test_bkg)
        #exit()
        
        ### Selecting train signal events with the same mass
        '''
        m_train_signal = X_train_signal[:, InputFeatures.index('mass')]
        X_train_signal_mass = X_train_signal[m_train_signal == mass]
        '''
        data_train_signal_mass = data_train_signal[data_train_signal['unscaledMass'] == unscaledMass]
        X_train_signal_mass = np.asarray(data_train_signal_mass[InputFeatures].values).astype(np.float32)
        wMC_train_signal_mass = np.array(data_train_signal_mass['weight'])

        ### Assigning the same mass value to train background events
        #X_train_bkg[:, InputFeatures.index('mass')] = np.full(len(X_train_bkg), UnscaledMass)
        data_train_bkg = data_train_bkg.assign(mass = np.full(len(data_train_bkg), scaledMass))
        X_train_bkg = np.asarray(data_train_bkg[InputFeatures].values).astype(np.float32)
        wMC_train_bkg = np.array(data_train_bkg['weight'])

        ### Prediction on signal and background
        yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg, batchSize)

        ### Drawing confusion matrix
        yhat_test_mass = np.concatenate((yhat_test_signal_mass, yhat_test_bkg_mass))
        y_test_mass = np.concatenate((np.ones(len(yhat_test_signal_mass)), np.zeros(len(yhat_test_bkg_mass))))
        wMC_test_mass = np.concatenate((wMC_test_signal_mass, wMC_test_bkg))

        TNR, FPR, FNR, TPR = DrawCM(yhat_test_mass, y_test_mass, wMC_test_mass, newOutputDir, unscaledMass, background, outputFileCommonName, jetCollection, analysis, channel, preselectionCuts, signal)
        newLogFile.write('\TNR (TN/N): ' + str(TNR) + '\nFPR (FP/N): ' + str(FPR) + '\FNR (FN/P): ' + str(FNR) + '\n TPR (TP/P): ' + str(TPR))

        ### Computing ROC AUC
        fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass, sample_weight = wMC_test_mass)
        #roc_auc = auc(fpr, tpr)
        #print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))
        #newLogFile.write('\nROC_AUC: ' + str(roc_auc))
        
        from scipy import integrate
        sorted_index = np.argsort(fpr)
        fpr_sorted =  np.array(fpr)[sorted_index]
        tpr_sorted = np.array(tpr)[sorted_index]
        auc = integrate.trapz(y = tpr_sorted, x = fpr_sorted)
        print(auc)
        
        ### Plotting ROC, background rejection and scores
        WP, bkgRejWP = DrawROCbkgRejectionScores(fpr_sorted, tpr_sorted, auc, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg)
        newLogFile.write('\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(bkgRejWP))
        
        bkgRej90.append(bkgRejWP[0])
        bkgRej94.append(bkgRejWP[1])
        bkgRej97.append(bkgRejWP[2])
        bkgRej99.append(bkgRejWP[3])
        bkgRejFile.write(str(unscaledMass) + ' ' + str(bkgRejWP[0]) + ' ' + str(bkgRejWP[1]) + ' ' + str(bkgRejWP[2]) + ' ' + str(bkgRejWP[3]) + '\n')
        '''
        bkgRej90Dict[unscaledMass].append(bkgRejWP[0])
        bkgRej94Dict[unscaledMass].append(bkgRejWP[1])
        bkgRej97Dict[unscaledMass].append(bkgRejWP[2])
        bkgRej99Dict[unscaledMass].append(bkgRejWP[3])
        print(bkgRej90Dict[unscaledMass])
        '''
        ### Closing the newLogFile
        newLogFile.close()
        print(Fore.GREEN + 'Saved ' + newLogFileName)
        '''
        if iLoop == (loop - 1):
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
            bkgRej90File.write(str(unscaledMass) + ' ')
            for rejValue in bkgRej90Dict[unscaledMass]:
                bkgRej90File.write(str(rejValue) + ' ')
            bkgRej90File.write('\n')
            bkgRej94File.write(str(unscaledMass) + ' ')
            for rejValue in bkgRej94Dict[unscaledMass]:
                bkgRej94File.write(str(rejValue) + ' ')
            bkgRej94File.write('\n')
            bkgRej97File.write(str(unscaledMass) + ' ')
            for rejValue in bkgRej97Dict[unscaledMass]:
                bkgRej97File.write(str(rejValue) + ' ')
            bkgRej97File.write('\n')
            bkgRej99File.write(str(unscaledMass) + ' ')
            for rejValue in bkgRej99Dict[unscaledMass]:
                bkgRej99File.write(str(rejValue) + ' ')
            bkgRej99File.write('\n')
        '''
    if (len(testMass) > 1):
        DrawRejectionVsMass(testMass, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName) 

bkgRejFile.close()
bkgRej90File.close()
bkgRej94File.close()
bkgRej97File.close()
bkgRej99File.close()


'''
end = time.time()
print("The time of execution of above program is :", end-start)
'''
