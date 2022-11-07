from Functions import *
from keras.utils.vis_utils import plot_model
import eli5
from eli5.permutation_importance import get_score_importances

### Setting a seed for reproducibility
#tf.random.set_seed(1234)

#blockPrint()

NN = 'PDNN'
batchSize = 2048
patienceValue = 5

### Reading the command line
tag, jetCollection, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass, doTrain, doTest, loop, hpOptimization, drawPlots = ReadArgParser()

drawPlots = True
originsBkgTest = list(background.split('_'))

### Reading the configuration file
ntuplePath, dfPath, InputFeatures = ReadConfig(tag, analysis, jetCollection, signal)
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/ggFVBF' + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/' + 'ggFsameStatAsVBF/'# + 'tmp/' # + '_fullStat/'
print(Fore.GREEN + 'Input files directory: ' + inputDir)
outputFileCommonName = NN + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background

### Creating the output directory and the logFile
outputDir = inputDir + NN# + '_3'# + '/withDNNscore'# + '/test1'# + '/3layers'#'/ggFsameStatAsVBF'# + '/withDNNscore' #'/DNNScore_Z'# + '/' + preselectionCuts # + '_halfStat'
print(format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

logFileName = outputDir + '/logFile_' + outputFileCommonName + '.txt'
logFile = open(logFileName, 'w')
logInfo = ''
logString = WriteLogFile(tag, ntuplePath, InputFeatures, inputDir, hpOptimization, doTrain, doTest, validationFraction, batchSize, patienceValue)
logFile.write(logString)
logInfo += logString

### Loading input data
data_train, data_test, X_train, X_test, y_train, y_test, w_train, w_test = LoadData(inputDir, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures)
'''
### to make same stat as VBF
data_train_signal = data_train[data_train['isSignal'] == 1]
data_train_bkg = data_train[data_train['isSignal'] != 1]
data_train_signal = data_train_signal[:87156]
data_train_bkg = data_train_bkg[:35500]
data_train = pd.concat((data_train_signal, data_train_bkg), ignore_index = True)
data_train = ShufflingData(data_train)
X_train = np.array(data_train[InputFeatures].values).astype(np.float32)
y_train = np.array(data_train['isSignal'].values).astype(np.float32)
'''
### Writing dataframes composition to the log file
logString = '\nNumber of train events: ' + str(len(X_train)) + ' (' + str(int(sum(y_train))) + ' signal and ' + str(int(len(y_train) - sum(y_train))) + ' background), with MC weights: ' + str(sum(data_train['weight'])) + ' (' + str(sum(data_train[data_train['isSignal'] == 1]['weight'])) + ' signal and ' + str(sum(data_train[data_train['isSignal'] == 0]['weight'])) + ' background)\nNumber of test events: ' + str(len(X_test)) + ' (' + str(int(sum(y_test))) + ' signal and ' + str(int(len(y_test) - sum(y_test))) + ' background), with MC weights: ' + str(sum(data_test['weight'])) + ' (' + str(sum(data_test[data_test['isSignal'] == 1]['weight'])) + ' signal and ' + str(sum(data_test[data_test['isSignal'] == 0]['weight'])) + ' background)'
logFile.write(logString)
logInfo += logString

bkgRejFile = open(outputDir + '/BkgRejectionVsMass.txt', 'w')
#bkgRejFile.write('Background rejection obtained using the version of the software 02-feb-2022, lepton masses as input feature, WP = 0.90, 0.94, 0.97, 0.99\n')
'''
bkgRej90File = open(outputDir + '/BkgRejectionVsMassWP90.txt', 'w')
bkgRej94File = open(outputDir + '/BkgRejectionVsMassWP94.txt', 'w')
bkgRej97File = open(outputDir + '/BkgRejectionVsMassWP97.txt', 'w')
bkgRej99File = open(outputDir + '/BkgRejectionVsMassWP99.txt', 'w')
'''

bkgRej90 = []
bkgRej94 = []
bkgRej97 = []
bkgRej99 = []

### hp optimization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import keras_tuner
from keras_tuner.tuners import RandomSearch
if hpOptimization:

    def buildOptimizedModel(hp):
        model = tf.keras.Sequential()
        model.add(layers.Dense(units = hp.Int('units', min_value = 8, max_value = 200, step = 8), input_dim = len(InputFeatures), activation = 'relu'))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.3, step = 0.1)))
        for iLayer in range(hp.Int('layers', 1, 5)):
            model.add(tf.keras.layers.Dense(units = hp.Int('units_' + str(iLayer), 0, 5, step = 1), activation = 'relu'))
            model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(iLayer), 0, 0.3, step = 0.1)))
        model.add(Dense(1, activation = 'sigmoid'))
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
        model.compile(optimizer = optimizer, loss = 'binary_crossentropy', weighted_metrics = ['accuracy'])#, run_eagerly = True)
        #plot_model(model, to_file='untrainedModel.png', show_shapes=True, show_layer_names=True)        
        return model

    tuner = RandomSearch(
        buildOptimizedModel,
        objective = keras_tuner.Objective('val_accuracy', direction = 'max'),
        max_trials = 300, 
        executions_per_trial = 1,
        directory = outputDir + '/tunerTrials/',  
        overwrite = True
    )

    print(tuner.search_space_summary())
    stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patienceValue)
    tuner.search(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, validation_split = validationFraction, callbacks = [stop_early], batch_size = batchSize)
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)
    #print(model.summary())

    logFile.write('\n************************** HYPERPARAMETERS OPTIMIZATION RESULTS **************************')
    print('Number of nodes in layer number 0: ', tuner.get_best_hyperparameters()[0].get('units'))
    logFile.write('\nNumber of nodes in layer number 0: ' + str(tuner.get_best_hyperparameters()[0].get('units')))
    print('Dropout in layer number 0: ', tuner.get_best_hyperparameters()[0].get('dropout'))
    logFile.write('\nDropout in layer number 0: ' + str(tuner.get_best_hyperparameters()[0].get('dropout')))
    layersNumber = tuner.get_best_hyperparameters()[0].get('layers')
    print('Number of hidden layers: ', layersNumber)
    logFile.write('\nNumber of hidden layers: ' + str(layersNumber))
    for iLayer in range(1, layersNumber + 1):
        hp_nodes = 'units_' + str(iLayer - 1)
        hp_dropout = 'dropout_' + str(iLayer - 1)
        print('Number of nodes in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_nodes)))
        logFile.write('\nNumber of nodes in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_nodes)))
        print('Dropout in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_dropout)))
        logFile.write('\nDropout in hidden layer number ' + str(iLayer) + ': ' + str(tuner.get_best_hyperparameters()[0].get(hp_dropout)))
    print('Optimizer:', model.optimizer.get_config())
    logFile.write('\nOptimizer: ' + str(model.optimizer.get_config()))
    logFile.write('\n*******************************************************************************************')

'''
These dictionaries were created to evaluate the stability of the pDNN by making consecutive trainings and tests
They are not needed when performing only one training. Since this is the usual case these and the other lines involving these dictionaries are commented out. 
bkgRej90Dict = {}
bkgRej94Dict = {}
bkgRej97Dict = {}
bkgRej99Dict = {}
'''
for iLoop in range(loop):

    #enablePrint()
    #print(Fore.RED + 'loop: ' + str(iLoop))
    #blockPrint()

    #if iLoop == 0:
        #logString = '\nLoss: ' + Loss + '\nLearning rate: ' + str(learningRate) + '\nOptimizer: ' + str(Optimizer) + '\nweighted_metrics: ' + str(Metrics)
        #logFile.write(logString)
        #logInfo += logString
    
    if not doTrain:
        from keras.models import model_from_json
        ### Loading architecture and weights from file
        print(Fore.BLUE + 'Loading architecture and weights')
        architectureFileName = outputDir + '/architecture.json'
        with open(architectureFileName, 'r') as architectureFile:
            loadedModel = architectureFile.read()
        print(Fore.GREEN + 'Loaded ' + architectureFileName) 
        model = model_from_json(loadedModel)
        weightsFileName = outputDir + '/weights.h5'
        model.load_weights(weightsFileName)
        print(Fore.GREEN + 'Loaded ' + weightsFileName)
        '''
        ### Reading loss, optimizer and metrics from file
        Loss, Metrics, OptimizerLoaded, learningRate = ReadLossOptimizerMetrics(outputDir)
        if OptimizerLoaded == 'RMSprop':
            Optimizer = tf.keras.optimizers.RMSprop(learning_rate = learningRate)
        elif OptimizerLoaded == 'Adam':
            Optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)
        '''
        model.compile(loss = 'binary_crossentropy', weighted_metrics = ['accuracy']) #-> We don't care about the optimizer since we will only perform test, if loss and/or metric change save and then load them

    if doTrain:
        if not hpOptimization:
            ### Building and compiling the PDNN
            model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout)
            model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
            logString = '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of hidden layers: ' + str(numberOfLayers) + '\nDropout: ' + str(dropout) + '\nLoss: ' + Loss + '\nOptimizer: ' + str(Optimizer) + '\nLearning rate: ' + str(learningRate) + '\nMetrics: ' + str(Metrics)
            logFile.write(logString)
            logInfo += logString

        ### Training
        print(Fore.BLUE + 'Training the ' + NN)
        modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = batchSize, validation_split = validationFraction, verbose = 1, shuffle = False, callbacks = EarlyStopping(verbose = True, patience = patienceValue, monitor = 'val_loss', restore_best_weights = True))

        ### Saving to files
        SaveModel(model, outputDir, NN)
        plot_model(model, to_file='trainedModel.png', show_shapes=True, show_layer_names=True)

    if doTest:
        ### Evaluating the performance of the PDNN on the test sample and writing results to the log file
        print(Fore.BLUE + 'Evaluating the performance of the ' + NN)
        testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test, w_test, batchSize)

        if iLoop == 0:
            logString = '\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy)
            logFile.write(logString)
            logInfo += logString
            
    else:
        testLoss = testAccuracy = None

    #if doTrain and doTest:
    ### Drawing accuracy and loss
    if drawPlots and doTrain: ### THINK
        DrawLoss(modelMetricsHistory, testLoss, patienceValue, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)
        DrawAccuracy(modelMetricsHistory, testAccuracy, patienceValue, outputDir, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)

    if iLoop == 0:
        logFile.close()
        print(Fore.GREEN + 'Saved ' + logFileName)

    ### Features ranking
    def score(X, Y):
        y_pred = model.predict(X)
        fpr, tpr, thresholds = roc_curve(Y, y_pred)#, sample_weight = wMC_test_mass)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    

    deltasDict = {}
    '''
    base_score, score_decreases = get_score_importances(score, X_test, y_test)
    print('###### base_score ########')
    print(base_score)
    print('###### score_decreases ########')
    print(score_decreases)
    feature_importances = np.mean(score_decreases, axis = 0)
    print('###### feature_importances ########')
    print(feature_importances)
    '''
    if doTest == False:
        exit()
        
    ### Dividing signal from background
    data_test_signal = data_test[data_test['isSignal'] == 1]
    data_test_bkg = data_test[data_test['isSignal'] != 1]
    data_train_signal = data_train[data_train['isSignal'] == 1]
    data_train_bkg = data_train[data_train['isSignal'] != 1]

    ### Saving unscaled test signal mass values
    m_test_unscaled_signal = data_test_signal['unscaledMass']
    unscaledTestMassPointsList = list(dict.fromkeys(list(m_test_unscaled_signal)))

    ### If testMass = 'all', defining testMass as the list of test signal masses 
    if testMass == ['all']:
        testMass = list(int(item) for item in set(list(m_test_unscaled_signal)))
    else:
        testMass = list(int(item) for item in testMass)
    testMass.sort()

    for unscaledMass in testMass:

        if unscaledMass != 1000:# and unscaledMass != 600:
            continue

        ### Checking whether there are test events with the selected mass
        if unscaledMass not in unscaledTestMassPointsList:
            print(Fore.RED + 'No test signal with mass ' + str(unscaledMass))
            continue

        ### Creating new output directory and log file
        newOutputDir = outputDir + '/' + str(int(unscaledMass))
        print(format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
        newLogFileName = newOutputDir + '/logFile_' + outputFileCommonName + '_' + str(unscaledMass) + '.txt'
        newLogFile = open(newLogFileName, 'w')
        '''
        if (iLoop == 0 and loop > 1):
            bkgRej90Dict[unscaledMass] = []
            bkgRej94Dict[unscaledMass] = []
            bkgRej97Dict[unscaledMass] = []
            bkgRej99Dict[unscaledMass] = []
        '''
        ### Selecting only test signal events with the same mass value and saving them as an array
        data_test_signal_mass = data_test_signal[data_test_signal['unscaledMass'] == unscaledMass]
        scaledMass = list(set(list(data_test_signal_mass['mass'])))[0]
        X_test_signal_mass = np.asarray(data_test_signal_mass[InputFeatures].values).astype(np.float32)
        newLogFile.write(logInfo + '\nNumber of test signal events with mass ' + str(int(unscaledMass)) + ' GeV: ' + str(len(X_test_signal_mass)))
        wMC_test_signal_mass = np.array(data_test_signal_mass['weight'])

        ### Assigning the same mass value to test background events and saving them as an array
        data_test_bkg = data_test_bkg.assign(mass = np.full(len(data_test_bkg), scaledMass))
        X_test_bkg = np.asarray(data_test_bkg[InputFeatures].values).astype(np.float32)
        wMC_test_bkg = np.array(data_test_bkg['weight'])

        X_test_mass = np.concatenate((X_test_signal_mass, X_test_bkg))
        y_test_mass = np.concatenate((np.ones(len(X_test_signal_mass)), np.zeros(len(X_test_bkg)))) ### REDEFINED BELOW!!

        nIter = 100
        base_score, score_decreases = get_score_importances(score, X_test_mass, y_test_mass, n_iter = nIter) ### increase n_iter 
        print('###### base_score ########')
        print(base_score)
        print('###### score_decreases ########')
        print(score_decreases)
        feature_importances = np.mean(score_decreases, axis = 0)
        print('###### mean score_decreases #######')
        print(feature_importances)
        relative_feature_importances = feature_importances / base_score
        print('##### relative_feature_importances ######')
        deltasDict[unscaledMass] = relative_feature_importances
        print(deltasDict)
        for iFeature in range(len(InputFeatures)):
            feature = InputFeatures[iFeature]
            print(feature)
            histoValues = np.array([item[iFeature] for item in score_decreases])
            print('##### histoValues #####')
            print(histoValues)
            histoValues = histoValues / base_score
            print('##### histoValues / base_score #####')
            print(histoValues)
            legendText = 'Mean: ' + str(round(histoValues.mean(), 2)) + '\nstd: ' + str(round(histoValues.std(), 2)) + '\nEntries: ' + str(len(histoValues))
            plt.hist(histoValues, label = legendText)
            plt.xlabel('Relative AUC difference')
            plt.title(feature + ' - ' + signal + ' (1 TeV) ' + analysis + ' ' + channel)
            plt.savefig(outputDir + '/Histo_AUCdifference_' + feature + '_' + outputFileCommonName + '.png')
            plt.clf()
        continue 

        


        
        ### Selecting train signal events with the same mass
        data_train_signal_mass = data_train_signal[data_train_signal['unscaledMass'] == unscaledMass]
        X_train_signal_mass = np.asarray(data_train_signal_mass[InputFeatures].values).astype(np.float32)
        wMC_train_signal_mass = np.array(data_train_signal_mass['weight'])

        ### Assigning the same mass value to train background events
        data_train_bkg = data_train_bkg.assign(mass = np.full(len(data_train_bkg), scaledMass))
        X_train_bkg = np.asarray(data_train_bkg[InputFeatures].values).astype(np.float32)
        wMC_train_bkg = np.array(data_train_bkg['weight'])

        ### Prediction on signal and background
        yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg, batchSize)

        ### Drawing confusion matrix
        yhat_test_mass = np.concatenate((yhat_test_signal_mass, yhat_test_bkg_mass))
        y_test_mass = np.concatenate((np.ones(len(yhat_test_signal_mass)), np.zeros(len(yhat_test_bkg_mass))))
        wMC_test_mass = np.concatenate((wMC_test_signal_mass, wMC_test_bkg))

        TNR, FPR, FNR, TPR = DrawCM(yhat_test_mass, y_test_mass, wMC_test_mass, newOutputDir, unscaledMass, background, outputFileCommonName, jetCollection, analysis, channel, preselectionCuts, signal, drawPlots)
        newLogFile.write('\TNR (TN/N): ' + str(TNR) + '\nFPR (FP/N): ' + str(FPR) + '\FNR (FN/P): ' + str(FNR) + '\n TPR (TP/P): ' + str(TPR))

        ### Computing ROC AUC
        fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)#, sample_weight = wMC_test_mass)
        roc_auc = auc(fpr, tpr)
        print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))
        newLogFile.write('\nROC_AUC: ' + str(roc_auc))
        
        from scipy import integrate
        sorted_index = np.argsort(fpr)
        fpr_sorted =  np.array(fpr)[sorted_index]
        tpr_sorted = np.array(tpr)[sorted_index]
        #auc = integrate.trapz(y = tpr_sorted, x = fpr_sorted)
        #print(format(Fore.BLUE + 'ROC_AUC: ' + str(auc)))
        #newLogFile.write('\nROC_AUC: ' + str(auc))
        
        ### Plotting ROC, background rejection and scores
        #WP, bkgRejWP = DrawROCbkgRejectionScores(fpr_sorted, tpr_sorted, auc, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg)
        WP, bkgRejWP = DrawROCbkgRejectionScores(fpr, tpr, roc_auc, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg, drawPlots)
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

    fig, ax1 = plt.subplots(figsize = (13, 13))
    #ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
    xLabels = InputFeatures
    yLabels = list(deltasDict.keys())
    deltasList = []
    for yLabel in yLabels:
        aa = list(deltasDict[yLabel])
        deltasList.append(aa)
    print(deltasList)

    im = ax1.matshow(deltasList)#, vmin = -1, vmax = 1)
    plt.colorbar(im, ax = ax1)
    plt.xticks(range(len(InputFeatures)), InputFeatures, rotation = 'vertical')
    plt.yticks(range(len(yLabels)), yLabels)

    #for i in xrange(15):
    for i in range(len(xLabels)):
        #for j in xrange(15):
        for j in range(len(yLabels)):
            massValue = yLabels[j]
            #c = intersection_matrix[j,i]
            c = deltasDict[massValue][i]
            cDisplay = round(c, 4)
            #cDisplay = "{:.1e}".format(c)
            ax1.text(i, j, str(cDisplay), va='center', ha='center', fontsize = 8, color = 'r')
            
    plt.tight_layout()
    plt.savefig(outputDir + '/featuresRanking_' + outputFileCommonName + '.png')
    exit()
    if (len(testMass) > 1):
        DrawRejectionVsMass(testMass, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName) 

bkgRejFile.close()
'''
bkgRej90File.close()
bkgRej94File.close()
bkgRej97File.close()
bkgRej99File.close()
'''
