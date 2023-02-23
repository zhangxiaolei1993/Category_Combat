# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:36:35 2022
@author: dell
"""

#
from __future__ import absolute_import, print_function
import pandas as pd
import numpy as np
import numpy.linalg as la
import os
import random

os.chdir(r"E:\keyan\github\Category_Combat")
from standardize_across_features import Location_scale_model 
from bayesain_procession import Bayesian_process 


"""
Run Category_Combat to remove multicnter effects in radiomics data

Arguments
---------
dat : a pandas data frame or numpy array
    radiomics data to correct with shape = (features, samples)

covars : a pandas data frame Conditional design w/ shape = (samples, covariates)
    Center Grouping, Retaining Factors, Removing Factors
    Please refer to the example for specific format
    
batch_col : string indicating batch (scanner) column name in covars
    - e.g. manufacturer
    
reserves_cols: string or list of strings of categorical reserve variables to adjust for
    - e.g. male or female
    
remove_cols : string or list of strings of remove variables to adjust for
    - e.g. Reconstructed kernel, voxel size, tube current, tube voltage

event_col: string or list of strings of event results variables to adjust for
    - e.g. benign nodules, malignant nodules
    
mean_method: string or list of strings of 'Center' or 'Event' or 'feature mean'
    - e.g. 'Center'
    
eb : should Empirical Bayes be performed?
    - True by default

parametric : should parametric adjustements be performed?
    - True by default

mean_only : should only be the mean adjusted (no scaling)?
    - False by default
   
Returns
-------
- A numpy array with the same shape as `dat` which has now been Category_ComBat-harmonized
"""

os.chdir(r"E:\keyan\github\Category_Combat")

#parameter settings
parameter_dict = {}
parameter_dict['batch_col'] = ['center_name']
parameter_dict['reserve_cols'] = []
parameter_dict['remove_cols'] =[]
parameter_dict['event_col'] = ['Category']
parameter_dict['mean_method'] = ['event']
parameter_dict['discretization_coefficient'] = [0.5]
parameter_dict['eb'] = True
parameter_dict['parametric'] = True
parameter_dict['mean_only'] = False

#Read the parameter file of the radiomics data
covars =  pd.read_csv('.\data\\parameter.csv')
#Read the radiomics data
data = pd.read_csv('.\data\\radiomics_demo.csv')

#Data transpose
dat = data.T
 

############################################Location scale model
# Location scale model initialization
L_S_model = Location_scale_model(covars,parameter_dict,dat)
                                 
#Location-scale model data preprocessing
batch_levels, sample_per_batch,event_levels, sample_per_event,info_dict = L_S_model.data_procession()

#Design Matrix
design = L_S_model.make_design_matrix(None)

#Location-scale model parameter estimation
grand_mean,stand_mean,var_pooled,B_hat,betas_event \
     = L_S_model.Estimated_location_scale( design, info_dict) 
         
#Location-scale model normalization
s_data, s_mean = L_S_model.standardize_across_features(
               design, info_dict, stand_mean,var_pooled,B_hat)


###########################################Bayesian process
#Bayesian model initialization
L_S_bayesian = Bayesian_process(covars,parameter_dict,dat)

#Bayesian prior distribution parameter estimation
LS_dict = L_S_bayesian.fit_LS_model_and_find_priors(
            s_data, design, info_dict)

#Empirical Bayesian Posterior Distribution Parameter Estimation
gamma_star, delta_star = L_S_bayesian.find_parametric_adjustments(s_data,
                        LS_dict, info_dict)

#Non-empirical Bayesian Posterior Distribution Parameter Estimation
# gamma_star, delta_star = L_S_bayesian.find_non_parametric_adjustments(s_data,
#                         LS_dict, info_dict)

#Bayesian adjustment process
bayesdata = L_S_bayesian.adjust_data_final(s_data, design, gamma_star, delta_star,
                                s_mean, var_pooled, info_dict)















