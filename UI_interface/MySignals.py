# -*- coding: UTF-8 -*-
'''
@Time    : 2023/3/20 11:17
@Author  : 魏林栋
@Site    : 
@File    : MySignals.py
@Software: PyCharm
'''
from PySide2.QtCore import Signal, QObject

class MySignals(QObject):
    print_signal = Signal(str)
    pass
