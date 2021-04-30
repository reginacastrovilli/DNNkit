from Functions import *

plot = True
NN = 'DNN'

### Reading the command line
dropout, analysis, channel, trainingFraction, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction = ReadArgParser()

### Reading the configuration file
dfPath, modelPath, InputFeatures = ReadConfig(analysis)

### Loading data and creating input arrays
X_train, y_train, X_test, y_test = LoadDataCreateArrays(dfPath, analysis, channel, InputFeatures)

### Building the DNN
n_dim = X_train.shape[1] - 1 # mass won't be given as input to the DNN

model = BuildDNN(n_dim, numberOfNodes, numberOfLayers, dropout)

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### Dividing signal from background
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

X_test_signal = X_test[y_test == 1]
X_test_bkg = X_test[y_test != 1]
X_train_signal = X_train[y_train == 1]
X_train_bkg = X_train[y_train != 1]

### Saving mass values
m_train_signal = []
for event in X_train_signal:
    m_train_signal.append(event[-1]) ### ONLY IF 'DSID' IS THE LAST VARIABLE IN InputFeatures(Merged/Resolved)
massPointsList = list(set(m_train_signal))
print('Mass points: ' + str(massPointsList))

m_test_signal = []
for event in X_test_signal:
    m_test_signal.append(event[-1])

for mass in massPointsList:

    ### Creating the output directory
    outputDir = modelPath + NN + '/' + analysis + '/' + channel + '/' + str(int(mass))
    print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))

    ### Creating the logFile
    logFileName = outputDir + '/logFile.txt'
    logFile = open(logFileName, 'w')
    
    ### Writing previous information to the logFile
    logFile.write('dfPath: ' + dfPath + '\nmodelPath: ' + modelPath + '\nInputFeatures: ' + str(InputFeatures) + '\nAnalysis: ' + analysis + '\nChannel: ' + channel + '\nTraining fraction: ' + str(trainingFraction) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nMass points list: ' + str(massPointsList) + '\nMass point analyzed: ' + str(mass))

    ### Selecting only signal events with the same mass value 
    X_train_signal_mass = X_train_signal[m_train_signal == mass]
    X_test_signal_mass = X_test_signal[m_test_signal == mass]

    ### Scaling and dropping the mass value
    transformer = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, X_train_signal_mass.shape[1] - 1))], remainder = 'drop')
    X_train_signal_mass = transformer.fit_transform(X_train_signal_mass)
    X_test_signal_mass = transformer.fit_transform(X_test_signal_mass)
    X_train_bkg = transformer.fit_transform(X_train_bkg)
    X_test_bkg = transformer.fit_transform(X_test_bkg)

    ### Weighting events
    #W_Train_mass = EventsWeight(X_train_signal_mass, X_train_bkg)
    #    logFile.write('\nNumber of events in the signal/background train samples: ' + str(X_train_bkg.shape[0]))
    #    logFile.write('\nNumber of events in the signal/background test samples: ' + str(X_test_bkg.shape[0]))
    W_train_signal_mass, W_train_bkg = EventsWeight(X_train_signal_mass, X_train_bkg)

    ### Creating new extended arrays by adding W_train and value 1 (0) to signal (bkg) events (this information is needed in order to shuffle data properly)
    X_train_signal_mass_ext = np.insert(X_train_signal_mass, X_train_signal_mass.shape[1], W_train_signal_mass, axis = 1)
    X_train_signal_mass_ext = np.insert(X_train_signal_mass_ext, X_train_signal_mass_ext.shape[1], 1, axis = 1)
    X_train_bkg_ext = np.insert(X_train_bkg, X_train_bkg.shape[1], W_train_bkg, axis = 1)
    X_train_bkg_ext = np.insert(X_train_bkg_ext, X_train_bkg_ext.shape[1], 0, axis = 1)
    X_test_signal_mass_ext = np.insert(X_test_signal_mass, X_test_signal_mass.shape[1], 1, axis = 1)
    X_test_bkg_ext = np.insert(X_test_bkg, X_test_bkg.shape[1], 0, axis = 1)

    ### Putting signal and background events back togheter
    X_train_mass_ext = np.concatenate((X_train_signal_mass_ext, X_train_bkg_ext), axis = 0) 
    X_test_mass_ext = np.concatenate((X_test_signal_mass_ext, X_test_bkg_ext), axis = 0)

    ### Shuffling data
    X_train_mass_ext = ShufflingData(X_train_mass_ext)
    X_test_mass_ext = ShufflingData(X_test_mass_ext)

    ### Extracting y_mass from X_mass_ext
    y_train_mass = X_train_mass_ext[:, X_train_mass_ext.shape[1] - 1]    
    y_test_mass = X_test_mass_ext[:, X_test_mass_ext.shape[1] - 1]    

    ### Deleting y_mass from X_mass
    X_train_mass_ext = np.delete(X_train_mass_ext, X_train_mass_ext.shape[1] - 1, axis = 1)
    X_test_mass = np.delete(X_test_mass_ext, X_test_mass_ext.shape[1] - 1, axis = 1)

    ### Extracting W_train_mass from X_train_mass_ext
    W_train_mass = X_train_mass_ext[:, X_train_mass_ext.shape[1] - 1]    

    ### Deleting W_train_mass from X_train_mass_ext
    X_train_mass = np.delete(X_train_mass_ext, X_train_mass_ext.shape[1] - 1, axis = 1)

    ### Training
    callbacks = [
        # If we don't have a decrease of the loss for 11 epochs, terminate training.
        EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')
    ]
    
    modelMetricsHistory = model.fit(X_train_mass, y_train_mass, sample_weight = W_train_mass, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, callbacks = callbacks)

    ### Evaluating the performance of the DNN
    testLoss, testAccuracy = EvaluatePerformance(model, X_test_mass, y_test_mass)

    ### Drawing training history
    if plot:
        DrawAccuracy(modelMetricsHistory, testAccuracy, outputDir, NN, mass)
        DrawLoss(modelMetricsHistory, testLoss, outputDir, NN, mass)

    ### Saving the model
    SaveModel(model, outputDir)

    ### Prediction on the full test sample
    yhat_test = model.predict(X_test_mass, batch_size = 2048)

    ### Plotting ROC
    fpr, tpr, thresholds = roc_curve(y_test_mass, yhat_test)
    roc_auc = auc(fpr, tpr)
    print(format(Fore.BLUE + 'ROC_AUC: ' + str(roc_auc)))

    if plot:
        DrawROC(fpr, tpr, outputDir, mass)

        ### Plotting confusion matrix
        DrawCM(yhat_test, y_test_mass, True, outputDir, mass)

    ### Saving performance parameters
    logFile.write('\nTestLoss: ' + str(testLoss) + '\nTestAccuracy: ' + str(testAccuracy) + '\nROC_AUC: ' + str(roc_auc))

    ### Prediction on signal and background separately
    yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)

    if plot:
        ### Plotting scores
        DrawScores(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass)
    
    ### Closing the logFile
    logFile.close()
    print('Saved ' + logFileName)
