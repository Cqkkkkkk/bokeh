from PyQt5.QtCore import QRect, reset, Qt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QAction, QPushButton, QSlider, QWidget, QLabel, QGridLayout, QMenuBar, QMenu, QStatusBar, QApplication
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
import os
import cv2
from LeRes.GenerateDepth import TestDepth

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.mode = 'simple'
        self.kernel = 'mean'
        # for mode simple
        self.kernel_size = 21
        self.threshold = 30
        # for mode depth-aware
        self.a = 0.05
        self.df = 50

        self.image_path = 'data/input/1002.jpg'
        self.depth_path = 'data/depth/1002-depth.png'
        self.mask_path = 'data/mask/1002-mask.png'
        self.result = None
        self.initUI()

    def initUI(self):
        # Initial settings
        self.resize(1100, 900)
        self.setWindowTitle('Main')
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName('centralwidget')

        self.initButton()
        self.initLabel()
        self.initGrapic()
        self.initMenu()
        self.initStatusBar()
        self.initSlider()

        self.show()

    def initSlider(self):
        # slider for kernel size
        self.sli_kernel_size = QSlider(Qt.Horizontal, self)
        self.sli_kernel_size.setGeometry(QRect(840, 520, 150, 20))
        self.sli_kernel_size.valueChanged[int].connect(self.func_change_kernel_size)
        self.sli_kernel_size.setTickInterval(10)
        self.sli_kernel_size.setToolTip('Set the size of blur kernel')

        self.sli_threshold = QSlider(Qt.Horizontal, self)
        self.sli_threshold.setGeometry(QRect(840, 550, 150, 20))
        self.sli_threshold.valueChanged[int].connect(self.func_change_threshold)
        self.sli_threshold.setTickInterval(10)
        self.sli_threshold.setToolTip("Set the threshold of the front scene's depth")

        self.sli_depth_a = QSlider(Qt.Horizontal, self)
        self.sli_depth_a.setGeometry(QRect(840, 520, 150, 20))
        self.sli_depth_a.valueChanged[int].connect(self.func_change_a)
        self.sli_depth_a.setTickInterval(10)
        self.sli_depth_a.setToolTip('Set the parameter a, which controls the iter rounds and depth-split scope')

        self.sli_depth_df = QSlider(Qt.Horizontal, self)
        self.sli_depth_df.setGeometry(QRect(840, 550, 150, 20))
        self.sli_depth_df.valueChanged[int].connect(self.func_change_df)
        self.sli_depth_df.setTickInterval(10)
        self.sli_depth_df.setToolTip('Set the focus depth df')

        self.sli_depth_a.setHidden(True)
        self.sli_depth_df.setHidden(True)

    

    def initGrapic(self):

        # Picture displays 
        gridLayoutWidget = QWidget(self.centralwidget)
        gridLayoutWidget.setGeometry(QRect(100, 70, 400, 300))
        gridLayoutWidget.setObjectName("gridLayoutWidget")
        gridLayout = QGridLayout(gridLayoutWidget)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.setObjectName("gridLayout")

        self.pic_original = QLabel()
        # pic_original.setPixmap(QPixmap('gui_assets/icon.png'))
        self.pic_original.setObjectName("label")
        self.pic_original.setScaledContents(True)
        self.pic_original.mousePressEvent = self.func_get_depth
        gridLayout.addWidget(self.pic_original, 0, 1, 1, 1)

        gridLayoutWidget_2 = QWidget(self.centralwidget)
        gridLayoutWidget_2.setGeometry(QRect(600, 70, 400, 300))
        gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        gridLayout_2 = QGridLayout(gridLayoutWidget_2)
        gridLayout_2.setContentsMargins(0, 0, 0, 0)
        gridLayout_2.setObjectName("gridLayout_2")

        self.pic_blured = QLabel()
        # pic_blured.setPixmap(QPixmap('gui_assets/icon.png'))
        self.pic_blured.setObjectName("label")
        self.pic_blured.setScaledContents(True)
        gridLayout_2.addWidget(self.pic_blured, 0, 1, 1, 1)

        gridLayoutWidget_3 = QWidget(self.centralwidget)
        gridLayoutWidget_3.setGeometry(QRect(100, 470, 400, 300))
        gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        gridLayout_3 = QGridLayout(gridLayoutWidget_3)
        gridLayout_3.setContentsMargins(0, 0, 0, 0)
        gridLayout_3.setObjectName("gridLayout_3")

        self.pic_depth = QLabel()
        # pic_blured.setPixmap(QPixmap('gui_assets/icon.png'))
        self.pic_depth.setObjectName("label")
        self.pic_depth.setScaledContents(True)
        self.pic_depth.mousePressEvent = self.func_get_depth
        gridLayout_3.addWidget(self.pic_depth, 0, 1, 1, 1)

        self.pic_depth.setPixmap(QPixmap(self.depth_path))
        self.pic_original.setPixmap(QPixmap(self.image_path))
        # self.pic_blured.setPixmap(QPixmap('data/input/1002.jpg'))

        self.setCentralWidget(self.centralwidget)


    def initLabel(self):
         # Text label 'original' and 'blur'
        label_original = QLabel(self.centralwidget)
        label_original.setGeometry(QRect(270, 40, 60, 16))
        label_original.setObjectName("Original")
        label_original.setText('Original')
        # label_original.setFont(QFont('思源宋体 CN', 12))

        label_blured = QLabel(self.centralwidget)
        label_blured.setGeometry(QRect(770, 40, 60, 16))
        label_blured.setObjectName("Blured")
        label_blured.setText('Blured')

        label_depth = QLabel(self.centralwidget)
        label_depth.setGeometry(QRect(270, 440, 60, 16))
        label_depth.setObjectName('Depth')
        label_depth.setText('Depth')

        self.label_slider_kernel_size = QLabel(self.centralwidget)
        self.label_slider_kernel_size.setGeometry(QRect(720, 497, 100, 20))
        self.label_slider_kernel_size.setObjectName('SliderKernelSizeLabel')
        self.label_slider_kernel_size.setText('Kernel size: {}'.format(self.kernel_size))

        self.label_slider_threshold = QLabel(self.centralwidget)
        self.label_slider_threshold.setGeometry(QRect(720, 527, 100, 20))
        self.label_slider_threshold.setObjectName('SliderThresholdLabel')
        self.label_slider_threshold.setText('Threshold: {}'.format(self.threshold))

        self.label_slider_a = QLabel(self.centralwidget)
        self.label_slider_a.setGeometry(QRect(720, 497, 100, 20))
        self.label_slider_a.setObjectName('SliderDepth-aLabel')
        self.label_slider_a.setText('a: {:.3f}'.format(self.a))

        self.label_slider_df = QLabel(self.centralwidget)
        self.label_slider_df.setGeometry(QRect(720, 527, 100, 20))
        self.label_slider_df.setObjectName('SliderDepth-dfLabel')
        self.label_slider_df.setText('df: {}'.format(self.df))
    
        self.label_slider_a.setHidden(True)
        self.label_slider_df.setHidden(True)


    def initMenu(self):

        # Menu bar
        menuBar = QMenuBar(self)
        menuBar.setGeometry(QRect(0, 0, 711, 21))
        menuBar.setObjectName("menuBar")
        menuFile = QMenu(menuBar)
        menuFile.setObjectName("menuFile")
        menuFile.setTitle('File')

        menuMode = QMenu(menuBar)
        menuMode.setObjectName("menuMode")
        menuMode.setTitle('Mode')

        menuKernel = QMenu(menuBar)
        menuKernel.setObjectName("menuKernel")
        menuKernel.setTitle('Kernel')

        self.setMenuBar(menuBar)

        actionExit = QAction()
        actionExit.setObjectName("actionExit")
        actionExit.setText('Exit')
        

        actionOpen = QAction(self)
        actionOpen.setObjectName("actionOpen")
        actionOpen.setText('Open')
        actionOpen.setToolTip('Select and open an image')
        actionOpen.triggered.connect(self.func_open_original)

        actionSave = QAction(self)
        actionSave.setObjectName('actionSave')
        actionSave.setText('Save')
        actionSave.setToolTip('Save the blured image')
        actionSave.triggered.connect(self.func_save_blured)


        actionSimple = QAction(self)
        actionSimple.setObjectName("actionSimple")
        actionSimple.setText('Simple')
        actionSimple.setToolTip('Switch to the Simple blur effect')
        actionSimple.triggered.connect(self.func_set_type_simple)

        actionDepth = QAction(self)
        actionDepth.setObjectName("actionDepth")
        actionDepth.setText('Depth')
        actionDepth.setToolTip('Switch to the Depth-aware blur effect')
        actionDepth.triggered.connect(self.func_set_type_depth)

        actionDNN = QAction(self)
        actionDNN.setObjectName("actionDNN")
        actionDNN.setText('DNN')
        actionDNN.setToolTip('Switch to the Deep Neural Network based bokeh effect')
        actionDNN.triggered.connect(self.func_set_type_dnn)

        self.actionMean = QAction(self)
        self.actionMean.setObjectName("actionMean")
        self.actionMean.setText('Mean')
        self.actionMean.setToolTip('Using mean kernel')
        self.actionMean.triggered.connect(self.func_set_kernel_mean)

        self.actionGauss = QAction(self)
        self.actionGauss.setObjectName("actionGauss")
        self.actionGauss.setText('Gauss')
        self.actionGauss.setToolTip('Using gauss kernel')
        self.actionGauss.triggered.connect(self.func_set_kernel_gauss)

        self.actionRadial = QAction(self)
        self.actionRadial.setObjectName("actionRadial")
        self.actionRadial.setText('Radial')
        self.actionRadial.setToolTip('Using radial kernel')
        self.actionRadial.triggered.connect(self.func_set_kernel_radial)

        self.actionTanh = QAction(self)
        self.actionTanh.setObjectName("actionTanh")
        self.actionTanh.setText('Tanh')
        self.actionTanh.setToolTip('Using tanh kernel')
        self.actionTanh.triggered.connect(self.func_set_kernel_tanh)

        menuFile.addAction(actionOpen)
        menuFile.addAction(actionSave)
        menuFile.addAction(actionExit)
        menuMode.addAction(actionSimple)
        menuMode.addAction(actionDepth)
        menuMode.addAction(actionDNN)
        menuKernel.addAction(self.actionMean)
        menuKernel.addAction(self.actionGauss)
        menuKernel.addAction(self.actionRadial)
        menuKernel.addAction(self.actionTanh)

        menuFile.setToolTipsVisible(True)
        menuMode.setToolTipsVisible(True)
        menuKernel.setToolTipsVisible(True)

        menuBar.addAction(menuFile.menuAction())
        menuBar.addAction(menuMode.menuAction())
        menuBar.addAction(menuKernel.menuAction())
        
        menuBar.setNativeMenuBar(False)

    def initStatusBar(self):
        # Status bar
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

    def initButton(self):
        pushButton = QPushButton(self.centralwidget)
        pushButton.setGeometry(QRect(595, 500, 100, 50))
        pushButton.setObjectName("pushButton")
        pushButton.setText('Convert')
        pushButton.clicked.connect(self.func_do_convert)

    def func_get_depth(self, event):
        if self.mode == 'depth':
            x = event.pos().x()
            y = event.pos().y()
            depth = QImage(self.depth_path)
            depth_norm = cv2.imread(self.depth_path).max()

            x *= depth.width() / 400
            y *= depth.height() / 300
            c = depth.pixel(x, y)
            
            cur_depth = QColor(c).getRgb()[0]
            cur_depth = int(cur_depth / depth_norm * 255)
            self.df = cur_depth
            self.label_slider_df.setText('df: {}'.format(cur_depth))
            self.statusbar.showMessage('Now setting focus depth df to {:.0f}'.format(cur_depth))
            

    def func_open_original(self):
        path, type_ = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        if path != '':
            input_dir, code = os.path.split(path)
            dir_ = input_dir[: -6]
            code = code[:code.find('.')]
            self.depth_path = dir_ + '/depth/' + '{}-depth.png'.format(code)
            self.mask_path = dir_ + '/mask/' + '{}-mask.png'.format(code)
            self.image_path = path
            image = QPixmap(self.image_path)
            self.pic_original.setPixmap(image)

            if not os.path.exists(self.depth_path):
                self.statusbar.showMessage('Cannot detect depth map in {}. Generating depth map'.format(self.depth_path))
                print('Cannot detect depth map in {}. Generating depth map'.format(self.depth_path))
                TestDepth(input_image=self.image_path, output_dir='./data/depth')

            depth = QPixmap(self.depth_path)
            self.pic_depth.setPixmap(depth)
        
    def func_save_blured(self):
        if self.result is not None: 
            path, type_ = QFileDialog.getSaveFileName(self, 'Open a file', '', 'All Files (*.*)')
            self.statusbar.showMessage('Blured image save to {}'.format(path)) 
            cv2.imwrite(path, self.result)
        else:
            self.statusbar.showMessage('No images to save!') 
        

    def func_change_threshold(self, value):
        self.threshold =  int(value / 100 * 255)
        self.label_slider_threshold.setText('Threshold: {}'.format(self.threshold))
        self.statusbar.showMessage('Current front image depth threshold: {}'.format(self.threshold))

    def func_change_kernel_size(self, value):
        self.kernel_size = value + 1
        self.label_slider_kernel_size.setText('Kernel size: {}'.format(self.kernel_size))
        self.statusbar.showMessage('Current kernel size: {}'.format(self.kernel_size))

    def func_change_a(self, value):
        self.a = value / 1000 + 1e-3
        self.label_slider_a.setText('a: {:.3f}'.format(self.a))
        self.statusbar.showMessage('Current a: {:.3f}'.format(self.a))

    def func_change_df(self, value):
        self.df = int(value / 100 * 255)
        self.label_slider_df.setText('df: {}'.format(self.df))
        self.statusbar.showMessage('Current df: {}'.format(self.df))


    def func_set_type_simple(self):
        self.statusbar.showMessage('Now switching the blur mode to Simple')
        self.mode = 'simple'
        self.change_visibility()
        
    def func_set_type_depth(self):
        self.statusbar.showMessage('Now switching the blur mode to Depth-aware')
        self.mode = 'depth'
        self.kernel = 'radial'
        self.change_visibility()
        


    def func_set_type_dnn(self):
        self.statusbar.showMessage('Now switching the blur mode to Deep Neural Networks')
        self.mode = 'dnn'
        self.change_visibility()

    def func_set_kernel_mean(self):
        self.statusbar.showMessage('Now switching the blur kernel to Mean')
        self.kernel = 'mean'
        
    def func_set_kernel_gauss(self):
        self.statusbar.showMessage('Now switching the blur kernel to Gauss')
        self.kernel = 'gauss'
            
    def func_set_kernel_radial(self):
        self.statusbar.showMessage('Now switching the blur kernel to Radial')
        self.kernel = 'radial'
            
    def func_set_kernel_tanh(self):
        self.statusbar.showMessage('Now switching the blur kernel to Tanh')
        self.kernel = 'tanh'

    def func_do_convert(self):
        raise NotImplementedError   

    def change_visibility(self):
        if self.mode != 'simple':
            self.sli_kernel_size.setHidden(True)
            self.sli_threshold.setHidden(True)
            self.label_slider_kernel_size.setHidden(True)
            self.label_slider_threshold.setHidden(True)
            self.actionMean.setVisible(False)
            self.actionGauss.setVisible(False)
        else:
            self.sli_kernel_size.setHidden(False)
            self.sli_threshold.setHidden(False)
            self.label_slider_kernel_size.setHidden(False)
            self.label_slider_threshold.setHidden(False)
            self.actionMean.setVisible(True)
            self.actionGauss.setVisible(True)

        if self.mode != 'depth':
            self.sli_depth_a.setHidden(True)
            self.sli_depth_df.setHidden(True)
            self.label_slider_a.setHidden(True)
            self.label_slider_df.setHidden(True)
        else:
            self.sli_depth_a.setHidden(False)
            self.sli_depth_df.setHidden(False)
            self.label_slider_a.setHidden(False)
            self.label_slider_df.setHidden(False)

        if self.mode == 'dnn':
            self.actionTanh.setVisible(False)
            self.actionRadial.setVisible(False)    