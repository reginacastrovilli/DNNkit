### Make sure that 'DSID' is the last input variable
from Functions import *
#from keras.models import load_model
#from sklearn.preprocessing import StandardScaler

plot = True
NN = 'PDNN'

### Reading the command line
analysis, channel, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, trainingFraction = ReadArgParser()

### Reading the configuration file
dfPathOld, modelPath, InputFeatures = ReadConfig(analysis)

### Creating the output directory
outputDir = 'test/' + NN + '/' + analysis + '/' + channel
print ('Output directory: ' + Fore.GREEN + outputDir, checkCreateDir(outputDir))

### Creating the logFile
logFileName = outputDir + '/logFile.txt'
logFile = open(logFileName, 'w')
logFile.write('InputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nTraining fraction: ' + str(trainingFraction))

### Looping over subdirectories in rootDir to load data
rootDir = 'NN_SplitData/'
counter = 0
X_Train_Collection = {}
X_Test_Collection = {}
y_Train_Collection = {}
y_Test_Collection = {}
X_Train_signal_Collection = {}
X_Test_signal_Collection = {}
X_Train_bkg_Collection = {}
X_Test_bkg_Collection = {}

for subdir, dirs, files in os.walk(rootDir):
    for dir in dirs:
        dfPath = os.path.join(subdir, dir)
        X_Train, X_Test, y_train, y_test, dfInputTrain, dfInputTest = LoadDataNew(dfPath, analysis, channel, InputFeatures)
        '''
        ### Scaling data
        X_Train = StandardScaler().fit_transform(X_Train)
        X_Test = StandardScaler().fit_transform(X_Test)
        '''

        ### Storing data into collections
        X_Train_Collection[int(dir)] = X_Train
        X_Test_Collection[int(dir)] = X_Test
        y_Train_Collection[int(dir)] = y_train
        y_Test_Collection[int(dir)] = y_test
        X_Train_signal_Collection[int(dir)] = X_Train[y_train == 1]
        X_Test_signal_Collection[int(dir)] = X_Test[y_test == 1]
        X_Train_bkg_Collection[int(dir)] = X_Train[y_train != 1]
        X_Test_bkg_Collection[int(dir)] = X_Test[y_test != 1]

        ###modificare 
        if (counter == 0):
            X_Train_Total = X_Train
            X_Test_Total = X_Test
            y_train_Total = y_train
            y_test_Total = y_test
            counter += 1
        else:
            X_Train_Total = np.concatenate((X_Train_Total, X_Train), axis = 0)
            X_Test_Total = np.concatenate((X_Test_Total, X_Test), axis = 0)
            y_train_Total = np.concatenate((y_train_Total, y_train), axis = 0)
            y_test_Total = np.concatenate((y_test_Total, y_test), axis = 0)

        logFile.write('\nLoaded: ' + dfInputTrain + ' -> ' + str(X_Train.shape[0]) + ' events\nLoaded: ' + dfInputTest + ' -> ' + str(X_Test.shape[0]) + ' events')
        print('Loaded: ' + Fore.GREEN + dfInputTrain)
        print('Loaded: ' + Fore.GREEN + dfInputTest)

### Weighting train events
signalEventsNumber, bkgEventsNumber, WTrainSignal, WTrainBkg, w_train = EventsWeightNew(y_train_Total)
logFile.write('\nNumber of train events: ' + str(X_Train_Total.shape[0]) + '\nNumber of test events: ' + str(X_Test_Total.shape[0]) + '\nNumber of signal train events: ' + str(signalEventsNumber) + ', number of background train events: ' + str(bkgEventsNumber) + ' -> weight for signal events: ' + str(WTrainSignal) + ', weight for background events: ' + str(WTrainBkg))

### Building the PDNN
n_dim = X_Train_Total.shape[1]

model = BuildDNN(n_dim, numberOfNodes, numberOfLayers, dropout)
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### Training the PDNN
callbacks = [
    # If we don't have a decrease of the loss for 11 epochs, terminate training.
    EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')#, ModelCheckpoint('model.hdf5', save_weights_only = False, monitor = 'val_loss', mode = 'min', verbose = True, save_best_only = True)
]

modelMetricsHistory = model.fit(X_Train_Total, y_train_Total, sample_weight = w_train, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, shuffle = True, verbose = 1, callbacks = callbacks)

### Evaluating the performance of the PDNN
testLoss, testAccuracy = EvaluatePerformance(model, X_Test_Total, y_test_Total) ###masse ordinate
logFile.write('\nTest loss: ' + str(testLoss) + '\nTest accuracy: ' + str(testAccuracy))
logFile.close()
print('Saved ' + logFileName)

if plot:
    ### Drawing training history
    DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN)
    DrawLoss(modelMetricsHistory, testLoss, outputDir, NN)

### Prediction on the full (signal + background) test sample for each mass 
for mass in X_Train_signal_Collection.keys():
        
    ### Creating new output directory
    newOutputDir = 'test/' + NN + '/' + analysis + '/' + channel + '/' + str(mass)
    print ('Output directory: ' + Fore.GREEN + newOutputDir, checkCreateDir(newOutputDir))
    
    ### Creating new log file
    newLogFileName = newOutputDir + '/logFile.txt'
    newLogFile = open(newLogFileName, 'w')
    
    ### Loading train/test signal/background samples from collections
    X_Train_signal_mass = X_Train_signal_Collection[mass]
    X_Test_signal_mass = X_Test_signal_Collection[mass]
    X_Test_bkg_mass = X_Test_bkg_Collection[mass]
    X_Train_bkg_mass = X_Train_bkg_Collection[mass]

    newLogFile.write('\nNumber of signal train events with mass ' + str(mass) + ' GeV: ' + str(X_Train_signal_mass.shape[0]))
    newLogFile.write('\nNumber of signal test events with mass ' + str(mass) + ' GeV: ' + str(X_Test_signal_mass.shape[0]))
    '''
    ### Saving the scaled train/test mass vaules
    for event in X_Train_signal_mass:  ###modificare
        scaledMassTrain = event[X_Train_signal_mass.shape[1] - 1]
    for event in X_Test_signal_mass:
        scaledMassTest = event[X_Test_signal_mass.shape[1] - 1]
    '''

    ### Saving the scaled train/test mass vaules
    for event in X_Train_signal_mass:  ###modificare
        scaledMass = event[X_Train_signal_mass.shape[1] - 1]
        
    ### Assigning the same train/test signal mass to background
    for bkg in X_Train_bkg_mass:
        bkg[X_Train_bkg_mass.shape[1] - 1] = scaledMass #Train
    for bkg in X_Test_bkg_mass:
        bkg[X_Test_bkg_mass.shape[1] - 1] = scaledMass #Test
        
    ### Creating a new extended test array by adding value 1 (0) to signal (background) events (this information is needed in order to shuffle data properly)
    X_Test_signal_mass_ext = np.insert(X_Test_signal_mass, X_Test_signal_mass.shape[1], 1, axis = 1)
    X_Test_bkg_mass_ext = np.insert(X_Test_bkg_mass, X_Test_bkg_mass.shape[1], 0, axis = 1)
    
    ### Putting signal and background events back together
    X_Test_mass_ext = np.concatenate((X_Test_signal_mass_ext, X_Test_bkg_mass_ext), axis = 0)

    ### Shuffling data
    X_Test_mass_ext = ShufflingData(X_Test_mass_ext)

    ### Extracting y_test_mass
    y_test_mass = X_Test_mass_ext[:, X_Test_mass_ext.shape[1] - 1]

    ### Deleting y_test_mass from X_Test_mass
    X_Test_mass = np.delete(X_Test_mass_ext, X_Test_mass_ext.shape[1] - 1, axis = 1)

    newLogFile.write('\nNumber of test events with mass ' + str(mass) + ' GeV: ' + str(X_Test_mass.shape[0]))
    
    ### Prediction
    yhat_test_mass = model.predict(X_Test_mass, batch_size = 2048)

    ### Evaluating ROC
    fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test_mass)
    roc_auc = auc(fpr, tpr)
    print(Fore.BLUE + 'AUC: ' + str(roc_auc))
    newLogFile.write('\nAUC: ' + str(roc_auc))

    if plot:
        ### Plotting ROC
        DrawROC(fpr, tpr, roc_auc, newOutputDir, mass)

        ### Plotting confusion matrix
        DrawCM(yhat_test_mass, y_test_mass, True, newOutputDir, mass)

    ### Prediction on signal and background separately
    yhat_train_signal_mass, yhat_train_bkg_mass, yhat_test_signal_mass, yhat_test_bkg_mass = PredictionSigBkg(model, X_Train_signal_mass, X_Train_bkg_mass, X_Test_signal_mass, X_Test_bkg_mass)

    if plot:
        ### Plotting scores
        DrawScores(yhat_train_signal_mass, yhat_test_signal_mass, yhat_train_bkg_mass, yhat_test_bkg_mass, newOutputDir, NN, mass)

    ### Closing the newLogFile
    newLogFile.close()
    print('Saved ' + newLogFileName)

