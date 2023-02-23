# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:38:33 2022

@author: zhang xiaolei
"""

from __future__ import absolute_import, print_function
import pandas as pd
import numpy as np
import numpy.linalg as la
import os



#one-hot help function
def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
        
    return Y
           
#Least Squares Parameter Estimation Function
def get_beta_with_nan(yy, mod):
    wh = np.isfinite(yy)
    mod = mod[wh,:]
    yy = yy[wh]
    B = np.dot(np.dot(la.inv(np.dot(mod.T, mod)), mod.T), yy.T)
 
    return B
  
#Design Matrix Adjustment Help Function     
def design_col_2_zero(name_cols,n_batch,tmp_design_col):

    tmp_design_col1 = tmp_design_col.copy()
    for i in name_cols:
        col = i+n_batch-1
        tmp_design_col1[:,col] = 0 
        
    return tmp_design_col1




































