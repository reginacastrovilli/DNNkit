from Functions import *
from keras.utils.vis_utils import plot_model
import eli5
from eli5.permutation_importance import get_score_importances
from keras import backend
#from keras.callbacks import ReduceLROnPlateau, Callback

### Setting a seed for reproducibility
#tf.random.set_seed(1234)

#blockPrint()

NN = 'PDNN'
batchSize = 2048
patienceValue = 5

### Reading the command line
tag, jetCollection, analysis, channel, preselectionCuts, background, trainingFraction, signal, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, testMass, doTrain, doTest, loop, doHpOptimization, drawPlots, trainSet, studyLearningRate = ReadArgParser()

drawPlots = True
doFeaturesRanking = False
originsBkgTest = list(background.split('_'))
doSameStatAsVBF = False
#studyLearningRate = False
doStatisticTest = False

### Reading the configuration file
ntuplePath, dfPath, InputFeatures = ReadConfig(tag, analysis, jetCollection, signal)
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/ggFVBF' + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/'
inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/'# + 'tmp/' # + '_fullStat/' <<<<<----- questo
#inputDir = dfPath + analysis + '/' + channel + '/' + preselectionCuts + '/' + signal + '/' + background + '/' + 'ggFsameStatAsVBF/'# + 'tmp/' # + '_fullStat/'
#print(Fore.GREEN + 'Input files directory: ' + inputDir)
outputFileCommonName = NN + '_' + jetCollection + '_' + analysis + '_' + channel + '_' + preselectionCuts + '_' + signal + '_' + background

### Creating the output directory and the logFile
#outputDir = inputDir + NN + '_trainSet' + str(trainSet)#0'# + '_3'# + '/withDNNscore'# + '/test1'# + '/3layers'#'/ggFsameStatAsVBF'# + '/withDNNscore' #'/DNNScore_Z'# + '/' + preselectionCuts # + '_halfStat'
outputDir = inputDir + NN + '_2layers48nodesSwish_epxpypz'
#outputDir = inputDir + NN + 'hpOptimization'
print(format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))#inputDir = outputDir ########## TOGLIERE!!!!
print(Fore.GREEN + 'Input files directory: ' + inputDir)

logFileName = outputDir + '/logFile_' + outputFileCommonName + '.txt'
logFile = open(logFileName, 'w')
logInfo = ''
logString = WriteLogFile(tag, ntuplePath, InputFeatures, inputDir, doHpOptimization, doTrain, doTest, validationFraction, batchSize, patienceValue)
logFile.write(logString)
logInfo += logString
if doStatisticTest:
    AUCfile = open(outputDir + '/AUCvalues.txt', 'w')
    AUCfile.write('AUC values in each of the 20 iterations\n')
    lossFile = open(outputDir + '/lossValues.txt', 'w')
    lossFile.write('Train loss minimum - validation loss minimum - train loss at the validation loss minimum (in each of the 20 iterations)\n')
    scoresFileName = outputDir + '/scoresValues.txt'
    scoresFile = open(scoresFileName, 'w')
    scoresFile.write('Scores of the 20 fixed events in each of the 20 iterations\n')
    
### Loading input data
#data_train, data_test, X_train, X_test, y_train, y_test, w_train, w_test = LoadData(inputDir, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures, trainSet)
data_train, data_test = LoadData(inputDir, tag, jetCollection, signal, analysis, channel, background, trainingFraction, preselectionCuts, InputFeatures, trainSet)

'''
### to make same stat as VBF
if doSameStatAsVBF:
    data_train, X_train, y_train = SameStatAsVBF(data_train)
'''
### Writing dataframes composition to the log file
logString = '\nNumber of train events: ' + str(data_train.shape[0]) + ' (' + str(data_train[data_train['isSignal'] == 1].shape[0]) + ' signal and ' + str(data_train[data_train['isSignal'] == 0].shape[0]) + ' background), with MC weights: ' + str(sum(data_train['weight'])) + ' (' + str(sum(data_train[data_train['isSignal'] == 1]['weight']))+ ' signal and ' + str(sum(data_train[data_train['isSignal'] == 0]['weight'])) + ' background)\nNumber of test events: ' + str(data_test.shape[0]) + ' (' + str(data_test[data_test['isSignal'] == 1].shape[0]) + ' signal and ' + str(data_test[data_test['isSignal'] == 1].shape[0]) + ' background), with MC weights: ' + str(sum(data_test['weight'])) + ' (' + str(sum(data_test[data_test['isSignal'] == 1]['weight'])) + ' signal and ' + str(sum(data_test[data_test['isSignal'] == 0]['weight'])) + ' background)'
logFile.write(logString)
logInfo += logString

if doHpOptimization:
    model, logString = HpOptimization(InputFeatures, patienceValue, X_train, y_train, w_train, numberOfEpochs, validationFraction, batchSize, outputDir)
    logFile.write(logString)
    logInfo += logString

else: 
    ### Building and compiling the PDNN
    model, Loss, Metrics, learningRate, Optimizer = BuildNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout, studyLearningRate)
    model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
    NNdiagramName = outputDir + '/trainedModel.png'
    #plot_model(model, to_file = NNdiagramName, show_shapes = True, show_layer_names = True)
    print(Fore.GREEN + 'Saved ' + NNdiagramName)
    logString = '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of hidden layers: ' + str(numberOfLayers) + '\nDropout: ' + str(dropout) + '\nLoss: ' + Loss + '\nOptimizer: ' + str(Optimizer) + '\nLearning rate: ' + str(learningRate) + '\nMetrics: ' + str(Metrics)
    logFile.write(logString)
    logInfo += logString
                
if studyLearningRate:
    patiences = [5]#[2, 5, 10, 15]
    print(Fore.RED + 'Training ' + str(len(patiences)) + ' pDNN with decreasing learning rate and different patience values')
    lr_list, loss_list, acc_list, = list(), list(), list()
    for iPatience in range(len(patiences)):
        iPatienceValue = patiences[iPatience]
        print(Fore.BLUE + 'NN number ' + str(iPatience) + ' out of ' + str(len(patiences) - 1) + ' -> patience = ' + str(iPatienceValue))
        modelMetricsHistory, lrm_rates = TrainNN(X_train, y_train, w_train, iPatienceValue, numberOfEpochs, batchSize, validationFraction, model, studyLearningRate)#, iLoop, loop)
        #modelMetricsHistory, lrm_rates = TrainNN(X_train, y_train, w_train, iPatienceValue, numberOfEpochs, batchSize, validationFraction, NN, model, Loss, Optimizer, Metrics, iLoop, loop)
        loss_list.append(modelMetricsHistory.history['loss'])
        acc_list.append(modelMetricsHistory.history['accuracy'])
        lr_list.append(lrm_rates)
        
    # plot learning rates
    plotHistory(patiences, lr_list, 'LearningRate', outputDir, outputFileCommonName)
    # plot loss
    plotHistory(patiences, loss_list, 'Loss', outputDir, outputFileCommonName)
    # plot accuracy
    plotHistory(patiences, acc_list, 'Accuracy', outputDir, outputFileCommonName)

for iLoop in range(loop):

    outputDirLoop = outputDir + '/loop_' + str(iLoop)
    print(format('Output directory: ' + Fore.GREEN + outputDirLoop), checkCreateDir(outputDirLoop))

    if not doTrain:
        #model = LoadNN(outputDir)
        model = LoadNN(outputDirLoop)

    if doTrain:
        data_train = scaleTrainDataset(data_train, inputDir, InputFeatures, outputDirLoop)
        '''
        if not doHpOptimization:
            ### Building and compiling the PDNN
            model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout)
            model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
            if iLoop == 0:
                logString = '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of hidden layers: ' + str(numberOfLayers) + '\nDropout: ' + str(dropout) + '\nLoss: ' + Loss + '\nOptimizer: ' + str(Optimizer) + '\nLearning rate: ' + str(learningRate) + '\nMetrics: ' + str(Metrics)
                logFile.write(logString)
                logInfo += logString
        '''

        '''
        if not doHpOptimization:
            ### Building and compiling the PDNN
            model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout)
            model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
            NNdiagramName = outputDirLoop + '/trainedModel.png'
            plot_model(model, to_file = NNdiagramName, show_shapes = True, show_layer_names = True)
            print(Fore.GREEN + 'Saved ' + NNdiagramName)
            if iLoop == 0:
                logString = '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of hidden layers: ' + str(numberOfLayers) + '\nDropout: ' + str(dropout) + '\nLoss: ' + Loss + '\nOptimizer: ' + str(Optimizer) + '\nLearning rate: ' + str(learningRate) + '\nMetrics: ' + str(Metrics)
                logFile.write(logString)
                logInfo += logString
        '''
        
        '''
            ### Training
            if not doHpOptimization:
            ### Building and compiling the PDNN
            model, Loss, Metrics, learningRate, Optimizer = BuildDNN(len(InputFeatures), numberOfNodes, numberOfLayers, dropout)
            model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics)
        '''
           
        #modelMetricsHistory, lrm.lrates = TrainNN(X_train, y_train, w_train, patienceValue, numberOfEpochs, batchSize, validationFraction, NN, model, Loss, Optimizer, Metrics, iLoop, loop)
        if not studyLearningRate:
            print(Fore.BLUE + 'Training the ' + NN + ' -- loop ' + str(iLoop) + ' out of ' + str(loop - 1))

            #model.compile(loss = Loss, optimizer = Optimizer, weighted_metrics = Metrics) ### here?
            #justcommented modelMetricsHistory = TrainNN(X_train, y_train, w_train, patienceValue, numberOfEpochs, batchSize, validationFraction, model, studyLearningRate)#, iLoop, loop)
            modelMetricsHistory = TrainNN(data_train[InputFeatures], data_train['isSignal'], data_train['train_weight'], patienceValue, numberOfEpochs, batchSize, validationFraction, model, studyLearningRate)#, iLoop, loop)

        if doStatisticTest:
            ### Writing best losses to file
            loss_hist = modelMetricsHistory.history['loss']
            val_loss_hist = modelMetricsHistory.history['val_loss']
            best_epoch = np.argmin(loss_hist)# + 1
            lossFile.write(str(np.min(loss_hist)) + ' ' + str(np.min(val_loss_hist)) + ' ' + str(loss_hist[best_epoch]) + '\n')

        ### Saving to files
        #model.load_weights('tmp/checkpoint/model.hdf5')
        SaveModel(model, outputDirLoop, NN)
        plot_model(model, to_file = 'trainedModel.png', show_shapes = True, show_layer_names = True)

    '''
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
    '''
    testLoss = testAccuracy = None
    #if doTrain and doTest:
    ### Drawing accuracy and loss
    if drawPlots and doTrain: ### THINK
        DrawLoss(modelMetricsHistory, testLoss, patienceValue, outputDirLoop, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)
        DrawAccuracy(modelMetricsHistory, testAccuracy, patienceValue, outputDirLoop, NN, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName)

    if iLoop == 0:
        logFile.close()
        print(Fore.GREEN + 'Saved ' + logFileName)


    if doTest == False:
        exit()

    if doFeaturesRanking:
        deltasDict = {}
        
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
        '''
        if unscaledMass != 1000:# and unscaledMass != 600:
            continue
        '''
        ### Checking whether there are test events with the selected mass
        if unscaledMass not in unscaledTestMassPointsList:
            print(Fore.RED + 'No test signal with mass ' + str(unscaledMass))
            continue

        ### Creating new output directory and log file
        newOutputDir = outputDirLoop + '/' + str(int(unscaledMass))
        print(format('Output directory: ' + Fore.GREEN + newOutputDir), checkCreateDir(newOutputDir))
        newLogFileName = newOutputDir + '/logFile_' + outputFileCommonName + '_' + str(unscaledMass) + '.txt'
        newLogFile = open(newLogFileName, 'w')

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

        if doFeaturesRanking:
            FeaturesRanking(X_test_signal_mass, X_test_bkg, deltasDict, InputFeatures, signal, analysis, channel, outputDir, outputFileCommonName, drawPlots)

        if doStatisticTest:
            if iLoop == 0:
                ### Creating dataset with just 20 events
                data_test_signal_mass_10 = data_test_signal_mass[:10]
                data_test_bkg_10 = data_test_bkg[:10]
                data_test_20 = pd.concat((data_test_signal_mass_10, data_test_bkg_10))
                #print(data_test_20)
                #print(data_test_20.shape)
                data_test_20.to_pickle(outputDir + '/20events_test.pkl')
                print(Fore.GREEN + 'Saved ' + outputDir + '/20events_test.pkl')
                #print(yhat_20)
                #print(len(yhat_20))
            else:
                data_test_20 = pd.read_pickle(outputDir + '/20events_test.pkl')

            yhat_20 = np.array(model.predict(data_test_20[InputFeatures], batch_size = batchSize))

            for iScore in range(len(yhat_20)):
                scoresFile.write(str(yhat_20[iScore][0]) + ' ')
                if iScore == (len(yhat_20) - 1):
                    scoresFile.write('\n')

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
        newLogFile.write('\nTNR (TN/N): ' + str(TNR) + '\nFPR (FP/N): ' + str(FPR) + '\nFNR (FN/P): ' + str(FNR) + '\nTPR (TP/P): ' + str(TPR))

        ### Computing ROC AUC
        fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)#, sample_weight = wMC_test_mass)
        roc_auc = auc(fpr, tpr)
        print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))
        newLogFile.write('\nROC_AUC: ' + str(roc_auc))
        if doStatisticTest:
            AUCfile.write(str(roc_auc) + '\n')

        '''
        from scipy import integrate
        sorted_index = np.argsort(fpr)
        fpr_sorted =  np.array(fpr)[sorted_index]
        tpr_sorted = np.array(tpr)[sorted_index]
        #auc = integrate.trapz(y = tpr_sorted, x = fpr_sorted)
        #print(format(Fore.BLUE + 'ROC_AUC: ' + str(auc)))
        #newLogFile.write('\nROC_AUC: ' + str(auc))
        #WP, bkgRejWP = DrawROCbkgRejectionScores(fpr_sorted, tpr_sorted, auc, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg)
        '''

        ### Plotting ROC, background rejection and scores
        WP, bkgRejWP = DrawROCbkgRejectionScores(fpr, tpr, roc_auc, newOutputDir, NN, unscaledMass, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName, yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, wMC_train_signal_mass, wMC_test_signal_mass, wMC_train_bkg, wMC_test_bkg, drawPlots)
        newLogFile.write('\nWorking points: ' + str(WP) + '\nBackground rejection at each working point: ' + str(bkgRejWP))
        
        ### Closing the newLogFile
        newLogFile.close()
        print(Fore.GREEN + 'Saved ' + newLogFileName)

        if drawPlots and doFeaturesRanking:
            PlotFeaturesRanking(InputFeatures, deltasDict, newOutputDir, outputFileCommonName)

        ### Plotting calibration curve on signal (train + test) scores
        #CalibrationCurves(wMC_test_signal_mass, wMC_train_signal_mass, yhat_test_signal_mass, yhat_train_signal_mass, wMC_test_bkg, wMC_train_bkg, yhat_test_bkg_mass, yhat_train_bkg_mass, unscaledMass, newOutputDir, outputFileCommonName)

        #WeightedDistributionComparison(data_train_signal_mass, data_train_bkg, data_test_signal_mass, data_test_bkg, yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass, 'lep1_eta')
        

    #if (len(testMass) > 1):
    #    DrawRejectionVsMass(testMass, WP, bkgRej90, bkgRej94, bkgRej97, bkgRej99, outputDir, jetCollection, analysis, channel, preselectionCuts, signal, background, outputFileCommonName) 

if doStatisticTest:
    AUCfile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + '/AUCvalues.txt')
    lossFile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + '/lossValues.txt')
    scoresFile.close()
    print(Fore.GREEN + 'Saved ' + outputDir + '/scoresValues.txt')
