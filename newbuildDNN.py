### Make sure that 'DSID' is the last input variable
from Functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

plot = True
NN = 'DNN'

### Reading the command line
analysis, channel, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, trainingFraction = ReadArgParser()

### Reading the configuration file
dfPathOld, modelPath, InputFeatures = ReadConfig(analysis)

### Looping over input subdirectories inside rootDir
rootDir = 'NN_SplitData/'

for subdir, dirs, files in os.walk(rootDir):
    for dir in dirs:
        dfPath = os.path.join(subdir, dir)
        mass = int(dir)

        ### Creating the output directory
        outputDir = 'test/' + NN + '/' + analysis + '/' + channel + '/' + str(int(mass))
        #outputDir = modelPath + NN + '/' + analysis + '/' + channel + '/' + str(int(mass))
        print ('Output directory: ' + Fore.GREEN + outputDir, checkCreateDir(outputDir))

        ### Creating the logFile
        logFileName = outputDir + '/logFile.txt'
        logFile = open(logFileName, 'w')
        logFile.write('InputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nTraining fraction: ' + str(trainingFraction))

        ### Loading input dataframe
        X_train, X_test, y_train, y_test, dfInputTrain, dfInputTest = LoadDataNew(dfPath, analysis, channel, InputFeatures)
        print('Loaded: ' + Fore.GREEN + dfInputTrain)
        print('Loaded: ' + Fore.GREEN + dfInputTest)

        logFile.write('\nLoaded: ' + dfInputTrain + ' -> ' + str(X_train.shape[0]) + ' events\nLoaded: ' + dfInputTest + ' -> ' + str(X_test.shape[0]) + ' events')

        ### Deleting DSID column
        X_Train = np.delete(X_Train, X_Train.shape[1] - 1, axis = 1)
        X_Test = np.delete(X_Test, X_Test.shape[1] - 1, axis = 1)

        '''
        ### Scaling and dropping the mass value
        transformer = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, X_train.shape[1] - 1))], remainder = 'drop')
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.fit_transform(X_test)
        '''
        ### Dividing signal from background
        X_train_signal = X_train[y_train == 1]
        X_train_bkg = X_train[y_train != 1]
        X_test_signal = X_test[y_test == 1]
        X_test_bkg = X_test[y_test != 1]

        logFile.write('\nNumber of signal test events: ' + str(X_test_signal.shape[0]) + '\nNumber of background test events: ' + str(X_test_bkg.shape[0]))

        ### Weighting events
        signalEventsNumber, bkgEventsNumber, WTrainSignal, WTrainBkg, w_train  = EventsWeightNew(y_train)
        logFile.write('\nNumber of signal train events: ' + str(int(signalEventsNumber)) + ', number of background train events: ' + str(int(bkgEventsNumber)) + ' -> weight for signal events: ' + str(WTrainSignal) + ', weight for background events: ' + str(WTrainBkg))

        ### Building the DNN
        n_dim = X_train.shape[1]
        model = BuildDNN(n_dim, numberOfNodes, numberOfLayers, dropout)
        model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

        ### Training the DNN
        callbacks = [
            # If we don't have a decrease of the loss for 11 epochs, terminate training.
            EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')
        ]
        modelMetricsHistory = model.fit(X_train, y_train, sample_weight = w_train, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, shuffle = True, verbose = 1, callbacks = callbacks)

        ### Evaluating the performance of the DNN
        testLoss, testAccuracy = EvaluatePerformance(model, X_test, y_test)
        logFile.write('\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy))

        ### Drawing training history
        if plot:
            DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, mass)
            DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, mass)

        ### Saving the model
        #SaveModel(model, outputDir)

        ### Prediction on the full test sample
        yhat_test = model.predict(X_test, batch_size = 2048)

        ### Evaluating ROC
        fpr, tpr, thresholds = roc_curve(y_test, yhat_test)
        roc_auc = auc(fpr, tpr)
        print(format(Fore.BLUE + 'AUC: ' + str(roc_auc)))
        logFile.write('\nAUC: ' + str(roc_auc))

        if plot:
            ### Plotting ROC
            DrawROC(fpr, tpr, roc_auc, outputDir, mass)

            ### Plotting confusion matrix
            DrawCM(yhat_test, y_test, True, outputDir, mass)

        ### Prediction on signal and background separately
        yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg = PredictionSigBkg(model, X_train_signal, X_train_bkg, X_test_signal, X_test_bkg)

        if plot:
            ### Plotting scores
            DrawScores(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass)
        
        ### Closing the logFile
        logFile.close()
        print('Saved ' + logFileName)
