from Functions import *

plot = True
NN = 'DNN'

### Reading the command line
analysis, channel, trainingFraction, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction = ReadArgParser()

### Reading the configuration file
dfPath, modelPath, InputFeatures = ReadConfig(analysis)

### Loading data and creating input arrays
X_train, y_train, X_test, y_test = LoadDataCreateArrays(dfPath, analysis, channel, InputFeatures)

### Building the DNN
n_dim = X_train.shape[1] - 1 # mass won't be given as input to the DNN

model = BuildDNN(n_dim, numberOfNodes, numberOfLayers)

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
    m_train_signal.append(event[-1])
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
    W_Train_mass = NewEventsCut(X_train_signal_mass, X_train_bkg)
    #    logFile.write('\nNumber of events in the signal/background train samples: ' + str(X_train_bkg.shape[0]))
    #    logFile.write('\nNumber of events in the signal/background test samples: ' + str(X_test_bkg.shape[0]))
    print(W_Train_mass)
    ### Putting signal and background events back togheter
    X_train_mass = np.concatenate((X_train_signal_mass, X_train_bkg), axis = 0) 
    X_test_mass = np.concatenate((X_test_signal_mass, X_test_bkg), axis = 0)

    ### Creating y_train_mass and y_test_mass with the same dimension of X_train_mass and X_test_mass respectively
    y_train_signal_mass = np.ones(len(X_train_signal_mass))
    y_test_signal_mass = np.ones(len(X_test_signal_mass))
    y_train_bkg = np.zeros(len(X_train_bkg))
    y_test_bkg = np.zeros(len(X_test_bkg))
    y_train_mass = np.concatenate((y_train_signal_mass, y_train_bkg), axis = 0)
    y_test_mass = np.concatenate((y_test_signal_mass, y_test_bkg), axis = 0)

    ### Training
    callbacks = [
        # If we don't have a decrease of the loss for 11 epochs, terminate training.
        EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')
    ]
    
    #modelMetricsHistory = model.fit(X_train_mass, y_train_mass, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, callbacks = callbacks)
    modelMetricsHistory = model.fit(X_train_mass, y_train_mass, sample_weight = W_Train_mass, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, callbacks = callbacks)

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
        '''yResult_test_cls = np.array([ int(round(x[0])) for x in yhat_test])
        cnf_matrix = confusion_matrix(y_test_mass, yResult_test_cls)
        DrawCM(cnf_matrix, True, outputDir, mass)
        '''
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
    exit()
