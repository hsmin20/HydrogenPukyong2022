import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QAction, QFileDialog, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QInputDialog, \
    QLineEdit, QComboBox, QMessageBox, QCheckBox, QProgressBar, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView, QDialogButtonBox, QDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import time
from sklearn.metrics import r2_score

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class PukyongMachineLearner:
    def __init__(self):
        self.modelLoaded = False

    def set(self, nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, callback):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(nnList)
        self.modelLoaded = True

    def isModelLoaded(self):
        return self.modelLoaded

    def fit(self, x_data, y_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=500)
            _callbacks.append(early_stopping)

        if self.splitPercentage > 0:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              validation_split=self.splitPercentage, verbose=self.verbose,
                                              callbacks=[self.callback])
        else:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              verbose=self.verbose, callbacks=_callbacks)

        return training_history

    def fitWithValidation(self, x_train_data, y_train_data, x_valid_data, y_valid_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping()
            _callbacks.append(early_stopping)

        training_history = self.model.fit(x_train_data, y_train_data, batch_size=self.batchSize, epochs=self.epoch,
                                          verbose=self.verbose, validation_data=(x_valid_data, y_valid_data),
                                          callbacks=_callbacks)

        return training_history

    def predict(self, x_data):
        y_predicted = self.model.predict(x_data)

        return y_predicted

    def createModel(self, nnList):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        model = tf.keras.Sequential()

        firstLayer = True
        for nn in nnList:
            noOfNeuron = nn[0]
            activationFunc = nn[1]
            if firstLayer:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc, input_shape=[noOfNeuron]))
                firstLayer = False
            else:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc))

        model.compile(loss='mse', optimizer=adamOpt)

        if self.verbose:
            model.summary()

        return model

    def saveModel(self, foldername):
        if self.modelLoaded == True:
            self.model.save(foldername)

    def saveModelJS(self, filename):
        if self.modelLoaded == True:
            tfjs.converters.save_keras_model(self.model, filename)

    def loadModel(self, foldername):
        self.model = keras.models.load_model(foldername)
        self.modelLoaded = True

    def showResult(self, y_data, training_history, y_predicted, sensor_name, height):
        # max_val = max(y_predicted)
        index_at_max = max(range(len(y_predicted)), key=y_predicted.__getitem__)
        index_at_zero = index_at_max
        for i in range(index_at_max, len(y_predicted)-1):
            val = y_predicted[i]
            val_next = y_predicted[i+1]
            if val >= 0 and val_next <=0:
                index_at_zero = i
                break;

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        title = sensor_name + height + ' / zeroIndexTime=' + str(index_at_zero * 0.000002)
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        axs[0].scatter(x_display, y_data, color="red", s=1)
        axs[0].plot(x_display, y_predicted, color='blue')
        axs[0].grid()

        lossarray = training_history.history['loss']
        axs[1].plot(lossarray, label='Loss')
        axs[1].grid()

        plt.show()

    def showResultValid(self, training_history, y_train_data, y_train_pred, y_valid_data, y_valid_pred,
                        y_test_data, y_test_pred):
        y_train_op = []
        y_train_imp = []
        for i in range(len(y_train_data)):
            y_train_op.append(y_train_data[i][0])
            y_train_imp.append(y_train_data[i][1])
        y_train_pred_op = []
        y_train_pred_imp = []
        for i in range(len(y_train_pred)):
            y_train_pred_op.append(y_train_pred[i][0])
            y_train_pred_imp.append(y_train_pred[i][1])

        r2TrainOp = r2_score(y_train_op, y_train_pred_op)
        r2TrainImp = r2_score(y_train_imp, y_train_pred_imp)

        y_valid_op = []
        y_valid_imp = []
        for i in range(len(y_valid_data)):
            y_valid_op.append(y_valid_data[i][0])
            y_valid_imp.append(y_valid_data[i][1])
        y_valid_pred_op = []
        y_valid_pred_imp = []
        for i in range(len(y_valid_pred)):
            y_valid_pred_op.append(y_valid_pred[i][0])
            y_valid_pred_imp.append(y_valid_pred[i][1])

        r2ValidOp = r2_score(y_valid_op, y_valid_pred_op)
        r2ValidImp = r2_score(y_valid_imp, y_valid_pred_imp)


        if len(y_test_data) > 0:
            y_test_op = []
            y_test_imp = []
            for i in range(len(y_test_data)):
                y_test_op.append(y_test_data[i][0])
                y_test_imp.append(y_test_data[i][1])
            y_test_pred_op = []
            y_test_pred_imp = []
            for i in range(len(y_test_pred)):
                y_test_pred_op.append(y_test_pred[i][0])
                y_test_pred_imp.append(y_test_pred[i][1])

            r2TestOp = r2_score(y_test_op, y_test_pred_op)
            r2TestImp = r2_score(y_test_imp, y_test_pred_imp)
        else:
            r2TestOp = 0
            r2TestImp = 0

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        title = 'Sensors' + 'Height'
        fig.suptitle(title)

        datasize = len(y_train_data)
        x_display_train = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display_train[j][0] = j

        datasize = len(y_valid_data)
        x_display_valid = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display_valid[j][0] = j

        datasize = len(y_test_data)
        x_display_test = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display_test[j][0] = j

        axs[0, 0].scatter(x_display_train, y_train_op, color="red", s=1)
        axs[0, 0].scatter(x_display_train, y_train_pred_op, color='blue', s=1)
        axs[0, 0].scatter(x_display_train, y_train_imp, color="green", s=1)
        axs[0, 0].scatter(x_display_train, y_train_pred_imp, color='black', s=1)
        title = f'Train Data R2 (Op = {r2TrainOp}, Imp = {r2TrainImp})'
        axs[0, 0].set_title(title)
        axs[0, 0].grid()

        axs[0, 1].scatter(x_display_valid, y_valid_op, color="red", s=1)
        axs[0, 1].scatter(x_display_valid, y_valid_pred_op, color='blue', s=1)
        axs[0, 1].scatter(x_display_valid, y_valid_imp, color="green", s=1)
        axs[0, 1].scatter(x_display_valid, y_valid_pred_imp, color='black', s=1)
        title = f'Validation Data R2 (Op = {r2ValidOp}, Imp = {r2ValidImp})'
        axs[0, 1].set_title(title)
        axs[0, 1].grid()

        axs[1, 0].scatter(x_display_test, y_test_op, color="red", s=1)
        axs[1, 0].scatter(x_display_test, y_test_pred_op, color='blue', s=1)
        axs[1, 0].scatter(x_display_test, y_test_imp, color="green", s=1)
        axs[1, 0].scatter(x_display_test, y_test_pred_imp, color='black', s=1)
        title = f'Test Data R2 (Op = {r2TestOp}, Imp = {r2TestImp})'
        axs[1, 0].set_title(title)
        axs[1, 0].grid()

        lossarray = training_history.history['loss']
        vallossarray = training_history.history['val_loss']
        axs[1, 1].plot(lossarray, label='train loss')
        axs[1, 1].plot(vallossarray, label='val loss')
        axs[1, 1].set_title('Loss')
        axs[1, 1].grid()

        plt.show()

class LayerDlg(QDialog):
    def __init__(self, unit='128', af='relu'):
        super().__init__()
        self.initUI(unit, af)

    def initUI(self, unit, af):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')

        label1 = QLabel('Units', self)
        self.tbUnits = QLineEdit(unit, self)
        self.tbUnits.resize(100, 40)

        label2 = QLabel('Activation f', self)
        self.cbActivation = QComboBox(self)
        self.cbActivation.addItem(af)
        self.cbActivation.addItem('swish')
        self.cbActivation.addItem('relu')
        self.cbActivation.addItem('selu')
        self.cbActivation.addItem('sigmoid')
        self.cbActivation.addItem('softmax')

        if af == 'linear':
            self.cbActivation.setEnabled(False)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.tbUnits, 0, 1)

        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.cbActivation, 1, 1)

        layout.addWidget(self.buttonBox, 2, 1)

        self.setLayout(layout)

class PukyongMLWindow(QMainWindow):
    N_FEATURE = 3
    COL_WIDTH_PER_ONE_SENSOR = 6

    def __init__(self):
        super().__init__()

        self.distArrayName = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
        self.distList = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]

        self.heightArrayName = ['H5.0m', 'H2.0m', 'H1.0m', 'H0.7m', 'H0.0m']
        self.heightList = [5.0, 2.0, 1.0, 0.7, 0.0]

        self.barrierLocName = ['2m', '5m', '8m']
        self.barrierLocList = [2.0, 5.0, 8.0]

        self.distArray2m = []
        self.heightArray2m = []
        self.cbArray2m = []

        self.distArray5m = []
        self.heightArray5m = []
        self.cbArray5m = []

        self.distArray8m = []
        self.heightArray8m = []
        self.cbArray8m = []

        self.initUI()

        self.indexijs2m = []
        self.indexijs5m = []
        self.indexijs8m = []
        self.southSensors2mOP = []
        self.southSensors5mOP = []
        self.southSensors8mOP = []
        self.southSensors2mImp = []
        self.southSensors5mImp = []
        self.southSensors8mImp = []
        self.dataLoaded = False
        self.modelLearner = PukyongMachineLearner()

    def initMenu(self):
        # Menu
        openNN = QAction(QIcon('open.png'), 'Open NN', self)
        openNN.setStatusTip('Open Neural Network Structure from a File')
        openNN.triggered.connect(self.showNNFileDialog)

        saveNN = QAction(QIcon('save.png'), 'Save NN', self)
        saveNN.setStatusTip('Save Neural Network Structure in a File')
        saveNN.triggered.connect(self.saveNNFileDialog)

        exitMenu = QAction(QIcon('exit.png'), 'Exit', self)
        exitMenu.setStatusTip('Exit')
        exitMenu.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openNN)
        fileMenu.addAction(saveNN)
        fileMenu.addSeparator()
        fileMenu.addAction(exitMenu)

        self.statusBar().showMessage('Welcome to Machine Learning for Pukyong CFD')
        self.central_widget = QWidget()  # define central widget
        self.setCentralWidget(self.central_widget)  # set QMainWindow.centralWidget

    def initCSVFileReader(self):
        layout = QHBoxLayout()

        fileLabel = QLabel('csv file (This program is for data)')
        self.editFile = QLineEdit('Please load /srv/MLData/pukyong_flatwall258m_horizontal_x_opimpulse.csv')
        self.editFile.setFixedWidth(700)
        openBtn = QPushButton('...')
        openBtn.clicked.connect(self.showFileDialog)

        layout.addWidget(fileLabel)
        layout.addWidget(self.editFile)
        layout.addWidget(openBtn)

        return layout

    def initSensor2m(self):
        layout = QGridLayout()

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        for i in range(cols):
            cbDist = QCheckBox(self.distArrayName[i])
            cbDist.setTristate()
            cbDist.stateChanged.connect(self.distClicked2m)
            self.distArray2m.append(cbDist)

        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            cbHeight.setTristate()
            cbHeight.stateChanged.connect(self.heightClicked2m)
            self.heightArray2m.append(cbHeight)

        for i in range(rows):
            col = []
            for j in range(cols):
                if j == 1:
                    col.append(QCheckBox('||'))
                else:
                    col.append(QCheckBox(''))
            self.cbArray2m.append(col)

        barrierPos = QLabel('Barrier 2m')
        layout.addWidget(barrierPos, 0, 0)

        for i in range(len(self.distArray2m)):
            cbDist = self.distArray2m[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(self.heightArray2m)):
            cbHeight = self.heightArray2m[i]
            layout.addWidget(cbHeight, i + 1, 0)

        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray2m[i][j]
                qcheckbox.setTristate()
                layout.addWidget(qcheckbox, i + 1, j + 1)

        return layout

    def initSensor5m(self):
        layout = QGridLayout()

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        for i in range(cols):
            cbDist = QCheckBox(self.distArrayName[i])
            cbDist.setTristate()
            cbDist.stateChanged.connect(self.distClicked5m)
            self.distArray5m.append(cbDist)

        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            cbHeight.setTristate()
            cbHeight.stateChanged.connect(self.heightClicked5m)
            self.heightArray5m.append(cbHeight)

        for i in range(rows):
            col = []
            for j in range(cols):
                if j == 4:
                    col.append(QCheckBox('||'))
                else:
                    col.append(QCheckBox(''))
            self.cbArray5m.append(col)

        barrierPos = QLabel('Barrier 5m')
        layout.addWidget(barrierPos, 0, 0)

        for i in range(len(self.distArray5m)):
            cbDist = self.distArray5m[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(self.heightArray5m)):
            cbHeight = self.heightArray5m[i]
            layout.addWidget(cbHeight, i + 1, 0)

        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray5m[i][j]
                qcheckbox.setTristate()
                layout.addWidget(qcheckbox, i + 1, j + 1)

        return layout

    def initSensor8m(self):
        layout = QGridLayout()

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        for i in range(cols):
            cbDist = QCheckBox(self.distArrayName[i])
            cbDist.setTristate()
            cbDist.stateChanged.connect(self.distClicked8m)
            self.distArray8m.append(cbDist)

        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            cbHeight.setTristate()
            cbHeight.stateChanged.connect(self.heightClicked8m)
            self.heightArray8m.append(cbHeight)

        for i in range(rows):
            col = []
            for j in range(cols):
                if j == 7:
                    col.append(QCheckBox('||'))
                else:
                    col.append(QCheckBox(''))
            self.cbArray8m.append(col)

        barrierPos = QLabel('Barrier 8m')
        layout.addWidget(barrierPos, 0, 0)

        for i in range(len(self.distArray8m)):
            cbDist = self.distArray8m[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(self.heightArray8m)):
            cbHeight = self.heightArray8m[i]
            layout.addWidget(cbHeight, i + 1, 0)

        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray8m[i][j]
                qcheckbox.setTristate()
                layout.addWidget(qcheckbox, i + 1, j + 1)

        return layout

    def initReadOption(self):
        layout = QHBoxLayout()

        qcheckboxAll = QCheckBox('Check All')
        qcheckboxAll.stateChanged.connect(self.checkAll)
        loadButton = QPushButton('Load Data')
        loadButton.clicked.connect(self.loadData)
        showOverPressureButton = QPushButton('Show Overpressure Graph')
        showOverPressureButton.clicked.connect(self.showOverpressureGraphs)
        showImpulseButton = QPushButton('Show Impulse Graph')
        showImpulseButton.clicked.connect(self.showImpulseGraphs)

        layout.addWidget(qcheckboxAll)
        layout.addWidget(loadButton)
        layout.addWidget(showOverPressureButton)
        layout.addWidget(showImpulseButton)

        return layout

    def initNNTable(self):
        layout = QGridLayout()

        # NN Table
        self.tableNNWidget = QTableWidget()
        self.tableNNWidget.setColumnCount(2)
        self.tableNNWidget.setHorizontalHeaderLabels(['Units', 'Activation'])

        # read default layers
        DEFAULT_LAYER_FILE = 'default3_2.nn'
        self.updateNNList(DEFAULT_LAYER_FILE)

        self.tableNNWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableNNWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableNNWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Button of NN
        btnAdd = QPushButton('Add')
        btnAdd.setToolTip('Add a Hidden Layer')
        btnAdd.clicked.connect(self.addLayer)
        btnEdit = QPushButton('Edit')
        btnEdit.setToolTip('Edit a Hidden Layer')
        btnEdit.clicked.connect(self.editLayer)
        btnRemove = QPushButton('Remove')
        btnRemove.setToolTip('Remove a Hidden Layer')
        btnRemove.clicked.connect(self.removeLayer)
        btnLoad = QPushButton('Load')
        btnLoad.setToolTip('Load a NN File')
        btnLoad.clicked.connect(self.showNNFileDialog)
        btnSave = QPushButton('Save')
        btnSave.setToolTip('Save a NN File')
        btnSave.clicked.connect(self.saveNNFileDialog)
        btnMakeDefault = QPushButton('Make default')
        btnMakeDefault.setToolTip('Make this as a default NN layer')
        btnMakeDefault.clicked.connect(self.makeDefaultNN)

        layout.addWidget(self.tableNNWidget, 0, 0, 9, 6)
        layout.addWidget(btnAdd, 9, 0)
        layout.addWidget(btnEdit, 9, 1)
        layout.addWidget(btnRemove, 9, 2)
        layout.addWidget(btnLoad, 9, 3)
        layout.addWidget(btnSave, 9, 4)
        layout.addWidget(btnMakeDefault, 9, 5)

        return layout

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('32')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('10000')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        lrLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editLR = QLineEdit('0.00001')
        self.editLR.setFixedWidth(100)
        self.cbVerbose = QCheckBox('Verbose')
        self.cbVerbose.setChecked(True)

        splitLabel = QLabel('Split for Validation (0 means no split-data for validation)')
        splitLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editSplit = QLineEdit('0.2')
        self.editSplit.setFixedWidth(100)
        self.cbSKLearn = QCheckBox('use sklearn split')
        self.cbSKLearn.setChecked(True)
        self.cbEarlyStop = QCheckBox('Use Early Stopping (validation data)')
        multiplierLabel = QLabel('Multiplier for Barrier Position (0 is Normalization)')
        multiplierLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editMultiplier = QLineEdit('0')
        self.editMultiplier.setFixedWidth(100)

        self.cbMinMax = QCheckBox('Use Min/Max of  All data')
        self.cbMinMax.setChecked(True)
        self.cbValidData = QCheckBox('Use Partially checked  sensors as validation')
        self.cbValidData.setChecked(True)
        self.cbValidData.stateChanged.connect(self.validDataClicked)
        self.epochPbar = QProgressBar()

        layout.addWidget(batchLabel, 0, 0, 1, 1)
        layout.addWidget(self.editBatch, 0, 1, 1, 1)
        layout.addWidget(epochLabel, 0, 2, 1, 1)
        layout.addWidget(self.editEpoch, 0, 3, 1, 1)
        layout.addWidget(lrLabel, 0, 4, 1, 1)
        layout.addWidget(self.editLR, 0, 5, 1, 1)
        layout.addWidget(self.cbVerbose, 0, 6, 1, 1)

        layout.addWidget(splitLabel, 1, 0, 1, 2)
        layout.addWidget(self.editSplit, 1, 2, 1, 1)
        layout.addWidget(self.cbSKLearn, 1, 3, 1, 1)
        layout.addWidget(self.cbEarlyStop, 1, 4, 1, 1)
        layout.addWidget(multiplierLabel, 1, 5, 1, 1)
        layout.addWidget(self.editMultiplier, 1, 6, 1, 1)

        layout.addWidget(self.cbMinMax, 2, 0, 1, 2)
        layout.addWidget(self.cbValidData, 2, 2, 1, 2)
        layout.addWidget(self.epochPbar, 2, 4, 1, 4)

        return layout

    def initCommand(self):
        layout = QGridLayout()

        mlWithDataBtn = QPushButton('ML with Data')
        mlWithDataBtn.clicked.connect(self.doMachineLearning)
        self.cbResume = QCheckBox('Resume Learning')
        # self.cbResume.setChecked(True)
        saveModelBtn = QPushButton('Save Model')
        saveModelBtn.clicked.connect(self.saveModel)
        loadModelBtn =  QPushButton('Load Model')
        loadModelBtn.clicked.connect(self.loadModel)
        self.cbH5Format = QCheckBox('H5')
        checkValBtn = QPushButton('Check Trained')
        checkValBtn.clicked.connect(self.checkVal)
        saveModelJSBtn = QPushButton('Save Model for JS')
        saveModelJSBtn.clicked.connect(self.saveModelJS)

        layout.addWidget(mlWithDataBtn, 0, 0, 1, 1)
        layout.addWidget(self.cbResume, 0, 1, 1, 1)
        layout.addWidget(saveModelBtn, 0, 2, 1, 1)
        layout.addWidget(loadModelBtn, 0, 3, 1, 1)
        layout.addWidget(self.cbH5Format, 0, 4, 1, 1)
        layout.addWidget(checkValBtn, 0, 5, 1, 1)
        layout.addWidget(saveModelJSBtn, 0, 6, 1, 1)

        return layout

    def initGridTable(self):
        layout = QGridLayout()

        self.tableGridWidget = QTableWidget()

        self.tableGridWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.tableGridWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.tableGridWidget.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tableGridWidget.setColumnCount(1)
        item = QTableWidgetItem('')
        self.tableGridWidget.setHorizontalHeaderItem(0, item)

        # Buttons
        btnAddDist = QPushButton('Add Distance')
        btnAddDist.setToolTip('Add a Distance')
        btnAddDist.clicked.connect(self.addDistance)
        btnAddHeight = QPushButton('Add Height')
        btnAddHeight.setToolTip('Add a Height')
        btnAddHeight.clicked.connect(self.addHeight)
        btnRemoveDist = QPushButton('Remove Distance')
        btnRemoveDist.setToolTip('Remove a Distance')
        btnRemoveDist.clicked.connect(self.removeDistance)
        btnRemoveHeight = QPushButton('Remove Height')
        btnRemoveHeight.setToolTip('Remove a Height')
        btnRemoveHeight.clicked.connect(self.removeHeight)
        btnLoadDistHeight = QPushButton('Load')
        btnLoadDistHeight.setToolTip('Load predefined Distance/Height structure')
        btnLoadDistHeight.clicked.connect(self.loadDistHeight)
        barrierPos = QLabel('Barrier Position(m)')
        barrierPos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBarrierPos = QLineEdit('3')
        self.editBarrierPos.setFixedWidth(100)
        predictBtn = QPushButton('Predict')
        predictBtn.clicked.connect(self.predict)

        layout.addWidget(self.tableGridWidget, 0, 0, 9, 8)
        layout.addWidget(btnAddDist, 9, 0)
        layout.addWidget(btnAddHeight, 9, 1)
        layout.addWidget(btnRemoveDist, 9, 2)
        layout.addWidget(btnRemoveHeight, 9, 3)
        layout.addWidget(btnLoadDistHeight, 9, 4)
        layout.addWidget(barrierPos, 9, 5)
        layout.addWidget(self.editBarrierPos, 9, 6)
        layout.addWidget(predictBtn,9, 7)

        return layout

    def initUI(self):
        self.setWindowTitle('Machine Learning for Pukyong CFD')
        self.setWindowIcon(QIcon('web.png'))

        self.initMenu()

        layout = QVBoxLayout()

        sensorLayout2m = self.initSensor2m()
        sensorLayout5m = self.initSensor5m()
        sensorLayout8m = self.initSensor8m()
        readOptLayout = self.initReadOption()
        fileLayout = self.initCSVFileReader()
        cmdLayout = self.initCommand()
        nnLayout = self.initNNTable()
        mlOptLayout = self.initMLOption()
        tableLayout = self.initGridTable()

        layout.addLayout(fileLayout)
        layout.addLayout(sensorLayout2m)
        layout.addLayout(sensorLayout5m)
        layout.addLayout(sensorLayout8m)
        layout.addLayout(readOptLayout)
        layout.addLayout(mlOptLayout)
        layout.addLayout(nnLayout)
        layout.addLayout(cmdLayout)
        layout.addLayout(tableLayout)

        self.centralWidget().setLayout(layout)

        self.resize(1200, 800)
        self.center()
        self.show()

    def showNNFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open NN file', './', filter="NN file (*.nn);;All files (*)")
        if fname[0] != '':
            self.updateNNList(fname[0])

    def updateNNList(self, filename):
        self.tableNNWidget.setRowCount(0)
        with open(filename, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                strAr = line.split(',')
                self.tableNNWidget.insertRow(count)
                self.tableNNWidget.setItem(count, 0, QTableWidgetItem(strAr[0]))
                self.tableNNWidget.setItem(count, 1, QTableWidgetItem(strAr[1].rstrip()))
                count += 1

    def saveNNFileDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save NN file', './', filter="NN file (*.nn)")
        if fname[0] != '':
            self.saveNNFile(fname[0])

    def makeDefaultNN(self):
        filename = 'default.nn'
        self.saveNNFile(filename)
        QMessageBox.information(self, 'Saved', 'Neural Network Default Layers are set')

    def saveNNFile(self, filename):
        with open(filename, "w") as f:
            count = self.tableNNWidget.rowCount()
            for row in range(count):
                unit = self.tableNNWidget.item(row, 0).text()
                af = self.tableNNWidget.item(row, 1).text()
                f.write(unit + "," + af + "\n")

    def getNNLayer(self):
        nnList = []
        count = self.tableNNWidget.rowCount()
        for row in range(count):
            unit = int(self.tableNNWidget.item(row, 0).text())
            af = self.tableNNWidget.item(row, 1).text()
            nnList.append((unit, af))

        return nnList

    def addLayer(self):
        dlg = LayerDlg()
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            size = self.tableNNWidget.rowCount()
            self.tableNNWidget.insertRow(size-1)
            self.tableNNWidget.setItem(size-1, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(size-1, 1, QTableWidgetItem(af))

    def editLayer(self):
        row = self.tableNNWidget.currentRow()
        if row == -1 or row == (self.tableNNWidget.rowCount() - 1):
            return

        unit = self.tableNNWidget.item(row, 0).text()
        af = self.tableNNWidget.item(row, 1).text()
        dlg = LayerDlg(unit, af)
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            self.tableNNWidget.setItem(row, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(row, 1, QTableWidgetItem(af))

    def removeLayer(self):
        row = self.tableNNWidget.currentRow()
        if row > 0 and row < (self.tableNNWidget.rowCount() - 1):
            self.tableNNWidget.removeRow(row)

    def addDistance(self):
        sDist, ok = QInputDialog.getText(self, 'Input Distance', 'Distance to add:')
        if ok:
            cc = self.tableGridWidget.columnCount()
            self.tableGridWidget.setColumnCount(cc + 1)
            item = QTableWidgetItem('S' + sDist + 'm')
            self.tableGridWidget.setHorizontalHeaderItem(cc, item)

    def addHeight(self):
        sHeight, ok = QInputDialog.getText(self, 'Input Height', 'Height to add:')
        if ok:
            rc = self.tableGridWidget.rowCount()
            self.tableGridWidget.setRowCount(rc + 1)
            item = QTableWidgetItem('H' + sHeight + 'm')
            self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def removeDistance(self):
        col = self.tableGridWidget.currentColumn()
        if col == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        if col == 0:
            QMessageBox.warning(self, 'Warning', 'First column cannot be removed')
            return

        self.tableGridWidget.removeColumn(col)

    def removeHeight(self):
        row = self.tableGridWidget.currentRow()
        if row == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        self.tableGridWidget.removeRow(row)

    def loadDistHeight(self):
        fname = QFileDialog.getOpenFileName(self, 'Open distance/height data file', '/srv/MLData',
                                            filter="CSV file (*.csv);;All files (*)")

        if fname[0] != '':
            with open(fname[0], "r") as f:
                self.tableGridWidget.setRowCount(0)
                self.tableGridWidget.setColumnCount(0)

                lines = f.readlines()

                distAr = lines[0].split(',')
                for sDist in distAr:
                    cc = self.tableGridWidget.columnCount()
                    self.tableGridWidget.setColumnCount(cc + 1)
                    item = QTableWidgetItem('S' + sDist + 'm')
                    self.tableGridWidget.setHorizontalHeaderItem(cc, item)

                heightAr = lines[1].split(',')
                for sHeight in heightAr:
                    rc = self.tableGridWidget.rowCount()
                    self.tableGridWidget.setRowCount(rc + 1)
                    item = QTableWidgetItem('H' + sHeight + 'm')
                    self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def distClicked2m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray2m)
        for i in range(rows):
            qcheckbox = self.cbArray2m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked2m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray2m[0])
        for i in range(cols):
            qcheckbox = self.cbArray2m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked5m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray5m)
        for i in range(rows):
            qcheckbox = self.cbArray5m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked5m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray5m[0])
        for i in range(cols):
            qcheckbox = self.cbArray5m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked8m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray8m)
        for i in range(rows):
            qcheckbox = self.cbArray8m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked8m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray8m[0])
        for i in range(cols):
            qcheckbox = self.cbArray8m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked(self, cbArray, state):
        for row in range(5):
            for i in range(25):
                qcheckbox = cbArray[row][i]
                if state == Qt.Unchecked:
                    qcheckbox.setCheckState(Qt.Unchecked)
                elif state == Qt.Checked:
                    qcheckbox.setCheckState(Qt.Checked)
                else:
                    qcheckbox.setCheckState(Qt.PartiallyChecked)


    def checkAll(self, state):
        self.heightClicked(self.cbArray2m, state)
        self.heightClicked(self.cbArray5m, state)
        self.heightClicked(self.cbArray8m, state)

    def validDataClicked(self, state):
        if state == Qt.Checked:
            self.editSplit.setEnabled(False)
            self.cbSKLearn.setEnabled(False)
        else:
            self.editSplit.setEnabled(True)
            self.cbSKLearn.setEnabled(True)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData', filter="CSV file (*.csv);;All files (*)")
        if fname[0]:
            self.editFile.setText(fname[0])
            self.df = pd.read_csv(fname[0], dtype=float)

    def loadData(self):
        filename = self.editFile.text()
        if filename == '' or filename.startswith('Please'):
            QMessageBox.about(self, 'Warining', 'No CSV Data')
            return

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        listSelected2m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray2m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected2m.append((i, j))

        listSelected5m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray5m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected5m.append((i, j))

        listSelected8m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray8m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected8m.append((i, j))

        if len(listSelected2m) == 0 and len(listSelected5m) == 0 and len(listSelected8m) == 0:
            QMessageBox.information(self, 'Warning', 'Select sensor(s) first..')
            return

        try:
            self.readCSV(listSelected2m, listSelected5m, listSelected8m)

        except ValueError:
            QMessageBox.information(self, 'Error', 'There is some error...')
            return

        self.dataLoaded = True
        QMessageBox.information(self, 'Done', 'Data is Loaded')

    def readCSV(self, listSelected2m, listSelected5m, listSelected8m):
        self.indexijs2m.clear()
        self.indexijs5m.clear()
        self.indexijs8m.clear()
        self.southSensors2mOP.clear()
        self.southSensors5mOP.clear()
        self.southSensors8mOP.clear()
        self.southSensors2mImp.clear()
        self.southSensors5mImp.clear()
        self.southSensors8mImp.clear()

        for index_ij in listSelected2m:
            sensorName, op, imp = self.getPressureData(index_ij, 1)
            self.indexijs2m.append(index_ij)
            self.southSensors2mOP.append(op)
            self.southSensors2mImp.append(imp)

        for index_ij in listSelected5m:
            sensorName, op, imp = self.getPressureData(index_ij, (1 + 125) + 1)
            self.indexijs5m.append(index_ij)
            self.southSensors5mOP.append(op)
            self.southSensors5mImp.append(imp)

        for index_ij in listSelected8m:
            sensorName, op, imp = self.getPressureData(index_ij, (1 + 125) * 2 + 1)
            self.indexijs8m.append(index_ij)
            self.southSensors8mOP.append(op)
            self.southSensors8mImp.append(imp)

    def showOverpressureGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        xdata = []
        ydata = []

        numSensors = len(self.southSensors2mOP)
        for index in range(numSensors):
            overpressure = self.southSensors2mOP[index]

            index_ij = self.indexijs2m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier2m'

            xdata.append(sensorName)
            ydata.append(overpressure)

        numSensors = len(self.southSensors5mOP)
        for index in range(numSensors):
            overpressure = self.southSensors5mOP[index]

            index_ij = self.indexijs5m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier5m'

            xdata.append(sensorName)
            ydata.append(overpressure)

        numSensors = len(self.southSensors8mOP)
        for index in range(numSensors):
            overpressure = self.southSensors8mOP[index]

            index_ij = self.indexijs8m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier8m'

            xdata.append(sensorName)
            ydata.append(overpressure)

        plt.plot(xdata, ydata, '-o')

        plt.title('Overpressure Graph')
        plt.xlabel('Sensors')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.xticks(rotation=60)
        plt.grid()
        plt.tight_layout()

        plt.show()

    def showImpulseGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        xdata = []
        ydata = []

        numSensors = len(self.southSensors2mImp)
        for index in range(numSensors):
            impulse = self.southSensors2mImp[index]

            index_ij = self.indexijs2m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier2m'

            xdata.append(sensorName)
            ydata.append(impulse)

        numSensors = len(self.southSensors5mImp)
        for index in range(numSensors):
            impulse = self.southSensors5mImp[index]

            index_ij = self.indexijs5m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier5m'

            xdata.append(sensorName)
            ydata.append(impulse)

        numSensors = len(self.southSensors8mImp)
        for index in range(numSensors):
            impulse = self.southSensors8mImp[index]

            index_ij = self.indexijs8m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier8m'

            xdata.append(sensorName)
            ydata.append(impulse)

        plt.bar(xdata, ydata)

        plt.title('Impulse Graph')
        plt.xlabel('Sensors')
        plt.ylabel('Impulse(kgm)/s')
        plt.legend(loc='upper right', markerscale=4.)
        plt.xticks(rotation=60)
        plt.grid()
        plt.tight_layout()

        plt.show()

    def getPressureData(self, index_ij, columnToSkip):
        i = index_ij[0]
        j = index_ij[1]
        len_height = len(self.heightArrayName)
        len_dist = len(self.distArrayName)
        col_no = columnToSkip + len_dist * (len_height - 1 - i) + j
        sensorName = self.distArrayName[j] + self.heightArrayName[i]

        op = self.df.values[0:1, col_no:col_no+1].flatten()[0]
        imp = self.df.values[1:2, col_no:col_no + 1].flatten()[0]

        return sensorName, op, imp

    def _prepareForMachineLearning(self):
        heightList_n, distList_n = self._normalize()

        # separate data to train & validation
        barrierPositionList = [ 2, 5, 8 ]
        # Normalized check
        multiplier = float(self.editMultiplier.text())
        for i in range(3):
            if multiplier < 0.1:
                barrierPositionList[i] = (barrierPositionList[i] - 2) / 6
            else:
                barrierPositionList[i] *= multiplier

        trainArray2m, trainIndex2m, testArray2m, testIndex2m = self.getArrayFromSensors(self.cbArray2m,
                                                                                          self.indexijs2m,
                                                                                          self.southSensors2mOP,
                                                                                          self.southSensors2mImp)
        trainArray5m, trainIndex5m, testArray5m, testIndex5m = self.getArrayFromSensors(self.cbArray5m,
                                                                                          self.indexijs5m,
                                                                                          self.southSensors5mOP,
                                                                                          self.southSensors5mImp)
        trainArray8m, trainIndex8m, testArray8m, testIndex8m = self.getArrayFromSensors(self.cbArray8m,
                                                                                          self.indexijs8m,
                                                                                          self.southSensors8mOP,
                                                                                          self.southSensors8mImp)

        trainArray = trainArray2m + trainArray5m + trainArray8m
        trainIndex = trainIndex2m + trainIndex5m + trainIndex8m
        trainBarrierPos = []
        for i in range(len(trainArray2m)):
            trainBarrierPos.append(barrierPositionList[0])
        for i in range(len(trainArray5m)):
            trainBarrierPos.append(barrierPositionList[1])
        for i in range(len(trainArray8m)):
            trainBarrierPos.append(barrierPositionList[2])

        testArray = testArray2m + testArray5m + testArray8m
        testIndex = testIndex2m + testIndex5m + testIndex8m
        testBarrierPos = []
        for i in range(len(testArray2m)):
            testBarrierPos.append(barrierPositionList[0])
        for i in range(len(testArray5m)):
            testBarrierPos.append(barrierPositionList[1])
        for i in range(len(testArray8m)):
            testBarrierPos.append(barrierPositionList[2])

        if len(trainArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_trainValid_data = np.zeros(shape=(len(trainArray), self.N_FEATURE))
        y_trainValid_data = np.zeros(shape=(len(trainArray) , 2))

        x_test_data = np.zeros(shape=(len(testArray), self.N_FEATURE))
        y_test_data = np.zeros(shape=(len(testArray), 2))

        # get maxp/minp from all of the training data
        if self.cbMinMax.isChecked() == True:
            maxop, minop, maximp, minimp = self.getMinMaxOverpresssureImpulseOfAllData()
        else:
            maxop, minop, maximp, minimp = self.getMinMaxOverpresssureImpulseOfLoadedData(trainArray)

        for i in range(len(trainArray)):
            index_ij = trainIndex[i]
            s_data = trainArray[i]
            barrierPos = trainBarrierPos[i]
            x_data_1, y_data_1 = self._prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxop, minop, maximp, minimp, barrierPos)
            x_trainValid_data[i] = x_data_1
            y_trainValid_data[i] = y_data_1

        for i in range(len(testArray)):
            index_ij = testIndex[i]
            s_data = testArray[i]
            barrierPos = testBarrierPos[i]
            x_data_1, y_data_1 = self._prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxop, minop, maximp, minimp, barrierPos)
            x_test_data[i] = x_data_1
            y_test_data[i] = y_data_1

        splitPercentage = float(self.editSplit.text())
        x_train_data, x_valid_data, y_train_data, y_valid_data = train_test_split(x_trainValid_data, y_trainValid_data,
                                                                                  test_size=splitPercentage,
                                                                                  random_state=42)

        return x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data

    def getMinMaxOverpresssureImpulseOfAllData(self):
        maxOverpressure = -100000
        minOverpressure = 100000
        maxImpulse = -100000
        minImpuse = 100000

        for i in range(len(self.distList)*3 + 3):
            if i == 0 or i == 26 or i == 52:
                continue

            one_op_data = self.df.values[0:1, i:i+1].flatten()[0]
            if one_op_data > maxOverpressure:
                maxOverpressure = one_op_data
            if one_op_data < minOverpressure:
                minOverpressure = one_op_data

            one_imp_data = self.df.values[1:2, i:i + 1].flatten()[0]
            if one_imp_data > maxImpulse:
                maxImpulse = one_imp_data
            if one_imp_data < minImpuse:
                minImpuse = one_imp_data

        return maxOverpressure, minOverpressure, maxImpulse, minImpuse

    def getMinMaxOverpresssureImpulseOfLoadedData(self, trainArray):
        maxOverpressure = -100000
        minOverpressure = 100000
        maxImpulse = -100000
        minImpuse = 100000

        numSensors = len(trainArray)
        for i in range(numSensors):
            op = trainArray[i][0]
            imp = trainArray[i][1]
            if op > maxOverpressure:
                maxOverpressure = op
            if imp < minOverpressure:
                minOverpressure = imp

        return maxOverpressure, minOverpressure, maxImpulse, minImpuse

    def doMachineLearning(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m) or \
                (not self.southSensors2mOP and not self.southSensors5mOP and not self.southSensors8mOP):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data = \
            self._prepareForMachineLearning()

        self._doMachineLearning(x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data)

        QApplication.restoreOverrideCursor()

    def saveData(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m) or \
                (not self.southSensors2m and not self.southSensors5m and not self.southSensors8m):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        numSensors = len(self.southSensors)
        for i in range(numSensors):
            index_ij = self.indexijs[i]
            t_data = self.time_data
            s_data = self.southSensors[i]

            suggestion = '/srv/MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'Time, ' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '\n'
                file.write(column1)

                numData = len(s_data)
                for j in range(numData):
                    line = str(t_data[j]) + ',' + str(s_data[j])
                    file.write(line)
                    file.write('\n')

                file.close()

                QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveDataAll(self):
        suggestion = '/srv/MLData/Changed.csv'
        # suggestion = '../MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
        if filename[0] != '':
            file = open(filename[0], 'w')

            column1 = 'Time, '
            numSensors = len(self.southSensors)
            for i in range(numSensors):
                index_ij = self.indexijs[i]
                column1 += self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]]
                if i != (numSensors - 1):
                    column1 += ','
            column1 += '\n'

            file.write(column1)

            t_data = self.time_data

            numData = len(t_data)
            for j in range(numData):
                line = str(t_data[j]) + ','

                for i in range(numSensors):
                    s_data = self.southSensors[i]
                    line += str(s_data[j])
                    if i != (numSensors - 1):
                        line += ','

                file.write(line)
                file.write('\n')

            file.close()

            QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveModel(self):
        suggestion = '/srv/MLData'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")
        if filename[0] != '':
            self.modelLearner.saveModel(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved.')

    def saveModelJS(self):
        suggestion = '/srv/MLData'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")
        if filename[0] != '':
            self.modelLearner.saveModelJS(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved for Javascript.')

    def loadModel(self):
        if self.cbH5Format.isChecked():
            fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData',
                                                filter="CSV file (*.h5);;All files (*)")
            if fname[0] != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.modelLearner.loadModel(fname[0])
                QApplication.restoreOverrideCursor()

                QMessageBox.information(self, 'Loaded', 'Model is loaded.')

        else:
            fname = QFileDialog.getExistingDirectory(self, 'Select Folder', "/srv/MLData",
                                                      QFileDialog.ShowDirsOnly)

            if fname != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.modelLearner.loadModel(fname)
                QApplication.restoreOverrideCursor()

                QMessageBox.information(self, 'Loaded', 'Model is loaded.')

    def checkVal(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m) or \
                (not self.southSensors2m and not self.southSensors5m and not self.southSensors8m):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        x_data, y_data, x_valid_data, y_valid_data = self._prepareForMachineLearning()
        y_predicted = self.modelLearner.predict(x_data)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        r2All = r2_score(y_data, y_predicted)
        title = f'LSTM Validation (R2 = {r2All})'

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_predicted, label='predicted', color="blue", s=1)
        plt.title(title)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def predict(self):
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        distCount = self.tableGridWidget.columnCount()
        heightCount = self.tableGridWidget.rowCount()
        if distCount < 1 or heightCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or height to predict')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        sDistHeightArray = []
        distHeightArray = []

        # put sensors in order of original Pukyong Excel file
        for j in range(heightCount):
            rownum = heightCount - (j + 1)
            height = self.tableGridWidget.verticalHeaderItem(rownum).text()
            heightOnly = height[1:len(height) - 1]

            for i in range(distCount):
                dist = self.tableGridWidget.horizontalHeaderItem(i).text()
                distOnly = dist[1:len(dist) - 1]

                distf = float(distOnly)
                heightf = float(heightOnly)

                sDistHeightArray.append(dist+height)
                distHeightArray.append((distf, heightf))

        maxop, minop, maximp, minimp = self.getMinMaxOverpresssureImpulseOfAllData()

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)

        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)

        sBarrierPosition = self.editBarrierPos.text()
        barrierList = sBarrierPosition.split(',')

        x_pred_data = np.zeros(shape=(1, self.N_FEATURE))
        y_array = []
        for sBarrier in barrierList:
            barrierPosition = float(sBarrier)


            for distHeight in distHeightArray:
                x_data = self._prepareOneSensorForPredict(distHeight, maxDist, minDist, maxHeight, minHeight, barrierPosition)
                x_pred_data[0] = x_data

                y_predicted = self.modelLearner.predict(x_pred_data)
                self.unnormalize(y_predicted, maxop, minop, maximp, minimp)

                y_array.append(y_predicted[0])

            QApplication.restoreOverrideCursor()

            # self.savePressureData(distHeightArray, y_array)

            self.showPredictionGraphs(sDistHeightArray, distHeightArray, y_array)

    def saveOPImpulse(self, resultArray, barrierPosition):
        # reply = QMessageBox.question(self, 'Message', 'Do you want to save overpressure and impulse to a file?',
        #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        reply = QMessageBox.Yes
        if reply == QMessageBox.Yes:
            sMultiplier = self.editMultiplier.text()

            suggestion = '/srv/MLData/opAndImpulse_pukyong_B' + str(barrierPosition) + 'm_' + sMultiplier + 'xB.csv'
            # filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
            #
            # if filename[0] != '':
            if suggestion != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                # file = open(filename[0], 'w')
                file = open(suggestion, 'w')

                column1 = 'distance,height,indexAtMax,overpressure,indexAtZero,impulse\n'
                file.write(column1)

                for col in resultArray:
                    file.write(col+'\n')

                QApplication.restoreOverrideCursor()

    def savePressureData(self, distHeightArray, y_array):
        reply = QMessageBox.question(self, 'Message', 'Do you want to save Pressure data to files?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/predicted_barrier' + str(barrierPosition) + 'm.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)

                file = open(filename[0], 'w')

                column1 = 'Time,Barrier_Dist,'
                for distHeight in distHeightArray:
                    distance = distHeight[0]
                    height = distHeight[1]
                    colname = 'S' + str(distance) + 'mH' + str(height) + 'm,'
                    column1 += colname

                column1 = column1[:-1]
                column1 += '\n'
                file.write(column1)

                dataSize = len(y_array[0])
                for index in range(dataSize):
                    time = self.time_data[index]
                    nextColumn = str(time) + ',' + self.editBarrierPos.text() + ','

                    for yindex in range(len(y_array)):
                        one_y = y_array[yindex]
                        onep = one_y[index][0]
                        nextColumn += str(onep) + ','

                    nextColumn = nextColumn[:-1]
                    nextColumn += '\n'
                    file.write(nextColumn)

                QApplication.restoreOverrideCursor()

    def unnormalize(self, data, maxop, minop, maximp, minimp):
        for i in range(len(data)):
            data[i][0] = data[i][0] * (maxop - minop) + minop
            data[i][1] = data[i][1] * (maximp - minimp) + minimp

    def _prepareOneSensorForPredict(self, distHeight, maxDist, minDist, maxHeight, minHeight, barrierPosition):
        distance = (distHeight[0] - minDist) / (maxDist - minDist)
        height = (distHeight[1] - minHeight) / (maxHeight - minHeight)
        # check for barrierPosition normalization
        multiplier = float(self.editMultiplier.text())
        if multiplier < 0.1:
            barrierPosition = (barrierPosition - 2) / 6
        else:
            barrierPosition *= multiplier

        feature_size = self.N_FEATURE  # pressure and height and distance and barrier position
        x_data = np.zeros(self.N_FEATURE)

        x_data[0] = height
        x_data[1] = distance
        x_data[2] = barrierPosition

        return x_data

    def showPredictionGraphs(self, sDistHeightArray, distHeightArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        x_display = []
        y_pred_op = []
        y_pred_imp = []
        for i in range(len(y_array)):
            y_pred_op.append(y_array[i][0])
            y_pred_imp.append(y_array[i][1])

            distHeight = distHeightArray[i]
            lab = sDistHeightArray[i]
            x_display.append(lab)

            distance = distHeight[0]
            height = distHeight[1]

        plt.figure()
        plt.scatter(x_display, y_pred_op, label='predicted overpressure', s=1)
        plt.scatter(x_display, y_pred_imp, label='predicted impulse', s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

    def checkDataGraph(self, sensorName, time_data, rawdata, filetered_data, data_label, iterNum):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ' (iter=' + str(iterNum) + ', impulse=' + format(impulse_filtered, '.4f') + ')'
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def checkDataGraph2(self, sensorName, time_data, rawdata, filetered_data, data_label, index_at_max, overpressure, index_at_zero):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ', impulse=' + format(impulse_filtered, '.4f') + ',indexMax=' + str(index_at_max) \
                      + ',Overpressure=' + str(overpressure) + ',indexZero=' + str(index_at_zero)
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def _prepareOneSensorData(self, height, dist, data, maxop, minop, maximp, minimp, barrierPos):
        op = data[0]
        imp = data[1]

        op_n = (op - minop) / (maxop - minop)
        imp_n = (imp - minimp) / (maximp - minimp)

        x_data = np.zeros(self.N_FEATURE)

        # fill the data
        x_data[0] = height
        x_data[1] = dist
        x_data[2] = barrierPos

        y_data = np.zeros(2)
        y_data[0] = op_n
        y_data[1] = imp_n

        return x_data, y_data

    def getArrayFromSensors(self, cbArray, indexijs, sensorsOP, sensorsImp):
        numSensors = len(sensorsOP)

        # separate data to train & validation
        trainArray = []
        trainIndex = []
        validArray = []
        validIndex = []

        useValidation = self.cbValidData.isChecked()
        if useValidation:
            for i in range(numSensors):
                indexij = indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                one_op_data = sensorsOP[i]
                one_imp_data = sensorsImp[i]

                if cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                    validArray.append([one_op_data, one_imp_data])
                    validIndex.append(indexij)
                elif cbArray[row_num][col_num].checkState() == Qt.Checked:
                    trainArray.append([one_op_data, one_imp_data])
                    trainIndex.append(indexij)
        else:
            for i in range(numSensors):
                indexij = indexijs[i]
                # row_num = indexij[0]
                one_op_data = sensorsOP[i]
                one_imp_data = sensorsImp[i]

                trainArray.append([one_op_data, one_imp_data])
                trainIndex.append(indexij)

        return trainArray, trainIndex, validArray, validIndex

    def _normalize(self):
        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)
        for j in range(len(heightList_n)):
            heightList_n[j] = (heightList_n[j] - minHeight) / (maxHeight - minHeight)

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)
        for j in range(len(distList_n)):
            distList_n[j] = (distList_n[j] - minDist) / (maxDist - minDist)

        return heightList_n, distList_n

    def _doMachineLearning(self, x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data):
        epoch = int(self.editEpoch.text())
        batchSize = int(self.editBatch.text())
        learningRate = float(self.editLR.text())
        splitPercentage = float(self.editSplit.text())
        earlyStopping = self.cbEarlyStop.isChecked()
        verb = self.cbVerbose.isChecked()

        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fitWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data)

        y_train_pred = self.modelLearner.predict(x_train_data)
        y_valid_pred = self.modelLearner.predict(x_valid_data)
        if len(x_test_data) > 0:
            y_test_pred = self.modelLearner.predict(x_test_data)
        else:
            y_test_pred = []

        self.modelLearner.showResultValid(training_history, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, y_test_data, y_test_pred)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TfCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.curStep = 1

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.setValue(self.curStep)
        self.curStep += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PukyongMLWindow()
    sys.exit(app.exec_())