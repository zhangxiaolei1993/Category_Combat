# -*- coding: UTF-8 -*-
'''
@Time    : 2023/4/11 11:52
@Author  : 魏林栋
@Site    : 
@File    : main.py
@Software: PyCharm
'''
from PySide2.QtWidgets import QApplication
from PySide2.QtGui import QIcon
from MainWindow import MainWindow
from MySignals import MySignals
import qtmodern.styles
import qtmodern.windows

class MainService:
    def __init__(self):
        self.app = QApplication([])
        qtmodern.styles.dark(self.app)
        self.MySignals = MySignals()
        self.MainWindow = MainWindow(signals=self.MySignals)
        self.win = qtmodern.windows.ModernWindow(self.MainWindow)
        self.win.show()
        self.app.aboutToQuit.connect(self.quit)

    def quit(self):
        '''程序关闭'''
        print('关闭程序！')
        self.win.close()

if __name__ == '__main__':
    app = MainService()
    app.app.exec_()