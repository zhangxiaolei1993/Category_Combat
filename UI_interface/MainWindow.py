# -*- coding: UTF-8 -*-
'''
@Time    : 2023/4/11 11:57
@Author  : 魏林栋
@Site    : 
@File    : MainWindow.py
@Software: PyCharm
'''
from PySide2.QtWidgets import QWidget, QFileDialog, QMessageBox
from PySide2.QtGui import QPixmap, QIcon
from ui.ui_MainWindow import Ui_Form
import os
import pandas as pd
from lib.Tools import Category_Combat

class MainWindow(QWidget, Ui_Form,):
    parameter_dict = {'batch_col': ['center_name'],
                      'reserve_cols': [],
                      'remove_cols': [],
                      'event_col': ['Category'],
                      'mean_method': ['event'],
                      'discretization_coefficient': [1.0],
                      'eb': True,
                      'parametric': True,
                      'mean_only': False}
    covars = None
    dat = None

    def __init__(self, parent=None, signals=None):
        super(MainWindow, self).__init__(parent)
        self.signals = signals
        self.setupUi(self)
        self.initUI()
        self.signals.print_signal.connect(self.printf)

    def printf(self, info):
        self.textBrowser.append(info)

    def initUI(self):
        self.setWindowIcon(QIcon('./texture/icon.png'))
        self.pushButton.clicked.connect(self.readCsv1)  # 导入文件（获取条件）
        self.pushButton_2.clicked.connect(self.readCsv2)  # （放射组学）
        self.pushButton_3.clicked.connect(self.run)  # 运行
        self.pushButton_4.clicked.connect(self.closeWindow)  # 退出
        self.comboBox.addItems(['event', 'center'])
        self.comboBox_2.addItems(['True', 'False'])
        self.comboBox_3.addItems(['True', 'False'])
        self.comboBox_4.addItems([str(i/10) for i in range(1, 11)])

    def showEvent(self, event):
        self.label_9.setPixmap(QPixmap('./texture/icon.png').scaled(self.label_9.width(), self.label_9.height()))  # 中间的图标

    def run(self):
        self.readLines()
        print('run')
        if self.covars is None:
            QMessageBox.critical(self, '错误', '请先获取条件！')
            return
        if self.dat is None:
            QMessageBox.critical(self, '错误', '请先导入放射组学数据！')
            return
        if self.parameter_dict:
            data_np = Category_Combat(self.covars, self.parameter_dict, self.dat, self.signals)
            df = pd.DataFrame(data_np)
            filePath, _ = QFileDialog.getSaveFileName(
                self,  # 父窗口对象
                "保存文件",  # 标题
                "./",  # 起始目录
                "类型 (*.csv)"  # 选择类型过滤项，过滤内容在括号中
            )
            if filePath:
                df.to_csv(filePath, index=False)

    def readCsv1(self):
        filePath = self.selectSingalFile()
        if os.path.isfile(filePath):
            self.covars = pd.read_csv(filePath)
  

    def readCsv2(self):
        filePath = self.selectSingalFile()
        if os.path.isfile(filePath):
            self.dat = pd.read_csv(filePath)  #
            # 数据转置
            self.dat = self.dat.T
            

    def selectSingalFile(self, title="选择CSV文件", path="./", type_="类型 (*.csv)"):
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            title,  # 标题
            path,  # 起始目录
            type_  # 选择类型过滤项，过滤内容在括号中
        )
        return filePath

    def readLines(self):
        if self.lineEdit.text():
            self.parameter_dict['batch_col'] = [self.lineEdit.text()]
        self.parameter_dict['reserve_cols'] = [self.lineEdit_2.text()] if self.lineEdit_2.text() else []
        self.parameter_dict['remove_cols'] = [self.lineEdit_3.text()] if self.lineEdit_3.text() else []
        if self.lineEdit_4.text():
            self.parameter_dict['event_col'] = [self.lineEdit_4.text()]
        self.parameter_dict['mean_method'] = self.comboBox.currentText()
        self.parameter_dict['discretization_coefficient'] = [float(self.comboBox_4.currentText())]
        self.parameter_dict['parametric'] = self.comboBox_2.currentText() == 'True'
        self.parameter_dict['mean_only'] = self.comboBox_3.currentText() == 'True'

    def closeWindow(self):
        self.close()