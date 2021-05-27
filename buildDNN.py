### Make sure that 'DSID' is the last input variable
from Functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

plot = True
NN = 'DNN'

### Reading the command line
analysis, channel, numberOfNodes, numberOfLayers, numberOfEpochs, validationFraction, dropout, trainingFraction = ReadArgParser()

### Reading the configuration file
dfPath, modelPath, InputFeatures = ReadConfig(analysis)

### Loading input dataframe
X_input, y_input, dfInput = LoadData(dfPath, analysis, channel, InputFeatures)

### Building the DNN
n_dim = X_input.shape[1] - 1 # mass won't be given as input to the DNN
model = BuildDNN(n_dim, numberOfNodes, numberOfLayers, dropout)
#model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

### EJS tests 2021-05-26, remove all this stuff
#import matplotlib
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = [7,7]
#plt.rcParams.update({'font.size': 16})
#bins = np.linspace(0, 10000, 100)
#plt.hist(X_input[:,1], bins = bins, histtype = 'stepfilled', density = True)
#ScoresPltName = 'tmp.png'
#plt.savefig(ScoresPltName)
#print('Saved ' + ScoresPltName)
#plt.clf()
#print(X_input[:,0])
#print(X_input.mean(axis=0).shape)
#print(X_input.mean(axis=0))   ### <= this for mean
#print(X_input.std(axis=0))   ### <= this for stdev
#end EJS tests, remove up to here

### Scaling (except for the mass value)
transformer = ColumnTransformer(transformers = [('name', StandardScaler(), slice(0, X_input.shape[1] - 1))], remainder = 'passthrough')
X_input = transformer.fit_transform(X_input)

### Dividing signal from background
X_signal = X_input[y_input == 1]
X_bkg = X_input[y_input != 1]

### Saving (unscaled) signal mass values
m_signal = X_signal[:, X_signal.shape[1] - 1]
massPointsList = list(set(m_signal))
print('Mass points: ' + str(massPointsList))

for mass in massPointsList:

    ### Creating the output directory
    outputDir = modelPath + NN + '/' + analysis + '/' + channel + '/' + str(int(mass))
    print (format('Output directory: ' + Fore.GREEN + outputDir), checkCreateDir(outputDir))
    
    ### Creating the logFile
    logFileName = outputDir + '/logFile.txt'
    logFile = open(logFileName, 'w')

    ### Writing previous information to the logFile
    logFile.write('Input dataframe: ' + dfInput + '\nNumber of input events: ' + str(X_input.shape[0]) + ' (' + str(X_signal.shape[0]) + ' signal and ' + str(X_bkg.shape[0]) + ' background)' + '\nInputFeatures: ' + str(InputFeatures) + '\nNumber of nodes: ' + str(numberOfNodes) + '\nNumber of layers: ' + str(numberOfLayers) + '\nNumber of epochs: ' + str(numberOfEpochs) + '\nValidation fraction: ' + str(validationFraction) + '\nDropout: ' + str(dropout) + '\nTraining fraction: ' + str(trainingFraction) + '\nMass points list: ' + str(massPointsList) + '\nMass point analyzed: ' + str(mass))

    ### Selecting only signal events with the same mass value 
    X_signal_mass = X_signal[m_signal == mass]
    logFile.write('\nNumber of signal events with the analyzed mass: ' + str(X_signal_mass.shape[0]))

    ### Creating new extended arrays by adding 1 for signal events and 0 for background events
    X_signal_mass_ext = np.insert(X_signal_mass, X_signal_mass.shape[1], 1, axis = 1)
    X_bkg_ext = np.insert(X_bkg, X_bkg.shape[1], 0, axis = 1)

    ### Putting signal and background events back together
    X_mass_ext = np.concatenate((X_signal_mass_ext, X_bkg_ext), axis = 0)

    ### Shuffling data
    X_mass_ext = ShufflingData(X_mass_ext)

    ### Splitting into test/train samples
    Ntrain_stop = int(round(X_mass_ext.shape[0] * trainingFraction))
    X_train_mass_ext = X_mass_ext[:Ntrain_stop]
    X_test_mass_ext = X_mass_ext[Ntrain_stop:]    

    logFile.write('\nNumber of train events: ' + str(X_train_mass_ext.shape[0]) + '\nNumber of test events: ' + str(X_test_mass_ext.shape[0]))

    ### Extracting y_mass from X_mass_ext
    y_train_mass = X_train_mass_ext[:, X_train_mass_ext.shape[1] - 1]
    y_test_mass = X_test_mass_ext[:, X_test_mass_ext.shape[1] - 1]

    ### Deleting y_mass and mass
    X_train_mass = np.delete(X_train_mass_ext, [X_train_mass_ext.shape[1] - 2, X_train_mass_ext.shape[1] - 1], axis = 1)
    X_test_mass = np.delete(X_test_mass_ext, [X_test_mass_ext.shape[1] - 2, X_test_mass_ext.shape[1] - 1], axis = 1)

    ### Dividing signal from background
    X_train_signal_mass = X_train_mass[y_train_mass == 1]
    X_train_bkg = X_train_mass[y_train_mass != 1]
    X_test_signal_mass = X_test_mass[y_test_mass == 1]
    X_test_bkg = X_test_mass[y_test_mass != 1]

    logFile.write('\nNumber of signal train events: ' + str(X_train_signal_mass.shape[0]) + '\nNumber of bkg train events: ' + str(X_train_bkg.shape[0]))

    ### Weighting events
    W_train_signal_mass, W_train_bkg = EventsWeight(X_train_signal_mass, X_train_bkg)
    logFile.write('\nWeight for signal train events: ' + str(W_train_signal_mass) + '\nWeight for background train events: ' + str(W_train_bkg))

    ### Creating array of weights
    w_train_mass = []
    for event in y_train_mass:
        if event == 1:
            w_train_mass.append(W_train_signal_mass)
        elif event == 0: 
            w_train_mass.append(W_train_bkg)
    w_train_mass = np.array(w_train_mass)
    
    ### Training
    callbacks = [
        # If we don't have a decrease of the loss for 11 epochs, terminate training.
        EarlyStopping(verbose = True, patience = 10, monitor = 'val_loss')
    ]
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    modelMetricsHistory = model.fit(X_train_mass, y_train_mass, sample_weight = w_train_mass, epochs = numberOfEpochs, batch_size = 2048, validation_split = validationFraction, verbose = 1, callbacks = callbacks)

    ### Evaluating the performance of the DNN
    testLoss, testAccuracy = EvaluatePerformance(model, X_test_mass, y_test_mass)
    logFile.write('\nTestLoss: ' + str(testLoss) + '\nTestAccuracy: ' + str(testAccuracy))

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
    print(format(Fore.BLUE + 'AUC: ' + str(roc_auc)))
    logFile.write('\nAUC: ' + str(roc_auc))

    if plot:
        DrawROC(fpr, tpr, roc_auc, outputDir, mass)

        ### Plotting confusion matrix
        DrawCM(yhat_test, y_test_mass, True, outputDir, mass)

    ###### Prediction on signal and background separately

    ### Dividing signal from background
    X_train_signal_mass = X_train_mass[y_train_mass == 1]
    X_train_bkg = X_train_mass[y_train_mass != 1]
    X_test_signal_mass = X_test_mass[y_test_mass == 1]
    X_test_bkg = X_test_mass[y_test_mass != 1]

    yhat_train_signal, yhat_train_bkg, yhat_test_signal, yhat_test_bkg = PredictionSigBkg(model, X_train_signal_mass, X_train_bkg, X_test_signal_mass, X_test_bkg)

    if plot:
        ### Plotting scores
        DrawScores(yhat_train_signal, yhat_test_signal, yhat_train_bkg, yhat_test_bkg, outputDir, NN, mass)
    
    ### Closing the logFile
    logFile.close()
    print('Saved ' + logFileName)
