# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:12:53 2022

@author: dell
"""



from __future__ import absolute_import, print_function
import os
import pandas as pd
import numpy as np
import numpy.linalg as la
import math


class Location_scale_model():
    
    def __init__(self,covars,parameter_dict,dat, signal):
        #parameter initialization
        self.signal = signal
        self.covars = covars
        self.dat = np.array(dat, dtype='float32')
        self.batch_col = parameter_dict['batch_col']
        self.reserve_cols = parameter_dict['reserve_cols']
        self.remove_cols = parameter_dict['remove_cols'] 
        self.event_col = parameter_dict['event_col']
        self.mean_method = parameter_dict['mean_method'] 
        self.discretization_coefficient = parameter_dict['discretization_coefficient'][0]
        self.eb = parameter_dict['eb']
        self.parametric = parameter_dict['parametric']
        self.mean_only = parameter_dict['mean_only']
                    
        #Parameter detection
        # if not isinstance(self.covars, pd.DataFrame):
        #     raise ValueError('covars must be pandas dataframe -> try: covars = pandas.DataFrame(covars)')

        # if not isinstance(self.reserve_cols, (list,tuple)):
        #     if self.reserve_cols is None:
        #         self.reserve_cols = []
        #         print("22222")
        #     else:
        #         self.reserve_cols = [reserve_cols]
        #         print("111111111")
           
        # if not isinstance(self.remove_cols, (list,tuple)):
        #     if self.remove_cols is None:
        #         self.remove_cols = []
        #     else:
        #         self.remove_cols = [remove_cols]
                
        # if not isinstance(self.event_col, (list,tuple)):
        #     if self.event_col is None:
        #         self.event_col = []
        #     else:
        #         self.event_col = [remove_col]
  
    
    #Parameters and data preprocessing        
    def data_procession(self, *args, **kwargs):
        
        """
        Returns center result labels 
        event result labels 
        and dictionary of parameters
        """
        
        #get parameter column name
        covar_labels = np.array(self.covars.columns)#print(np.array(covars.columns))
        
        self.covars = np.array(self.covars, dtype='object') 
        for i in range(self.covars.shape[-1]):#返回最后一个张亮
            try:
                self.covars[:,i] = self.covars[:,i].astype('float32')#最后一列并转化为浮点型
            except:
                pass
        
        #get the index of the parameter column name
        self.batch_col = np.where(covar_labels==self.batch_col)[0][0]
        self.event_col = [np.where(covar_labels==c_var)[0][0] for c_var in self.event_col]
        self.remove_cols = [np.where(covar_labels==c_var)[0][0] for c_var in self.remove_cols]
        self.reserve_cols = [np.where(covar_labels==n_var)[0][0] for n_var in self.reserve_cols]
        
        #Convert the parameter to an integer representation    
        for i in range(self.covars.shape[1]):    
            self.covars[:,i] = np.unique(self.covars[:,i],return_inverse=True)[-1]
         
        # create dictionary that stores batch info
        (batch_levels, sample_per_batch) = np.unique(self.covars[:,self.batch_col],return_counts=True)
        (event_levels, sample_per_event) = np.unique(self.covars[:,self.event_col],return_counts=True)
        self.event_levels = event_levels

        info_dict = {
            'batch_levels': batch_levels.astype('int'),
            'n_batch': len(batch_levels),
            'n_sample': int(self.covars.shape[0]),
            'sample_per_batch': sample_per_batch.astype('int'),
            'batch_info': [list(np.where(self.covars[:,self.batch_col]==idx)[0]) for idx in batch_levels]
        }
        print("Data preprocessing has been completed...")
        self.signal.print_signal.emit('Data preprocessing has been completed...')
        return batch_levels, sample_per_batch,event_levels, sample_per_event,info_dict
     
    
    #Make a Design Matrix
    def make_design_matrix(self, m_method):
        
        """
        Return Matrix containing the following parts:
            - matrix of batch variable (full)
            - matrix for reserve columns
            - matrix for remove columns
        """

    #Parameters are extracted from parameter dictionary      
        Y            = self.covars   
        batch_col    = self.batch_col
        remove_cols  = self.remove_cols
        reserve_cols = self.reserve_cols
        
        if m_method == "event":
            batch_col    = self.event_col

        
        hstack_list = []
        
    
        ### batch one-hot ###
        # convert batch column to integer in case it's string
        batch = np.unique(Y[:,batch_col],return_inverse=True)[-1]
        batch_onehot = self.to_categorical(batch, len(np.unique(batch)))
        hstack_list.append(batch_onehot)
        
        ### reserve categorical  ###
        for reserve_col in reserve_cols:
            reserve = np.array(Y[:,reserve_col],dtype='float32')
            reserve = reserve.reshape(reserve.shape[0],1)
            hstack_list.append(reserve)
    
        ### remove categorical  ###
        for remove_col in remove_cols:
            remove = np.array(Y[:,remove_col],dtype='float32')
            remove = remove.reshape(remove.shape[0],1)
            hstack_list.append(remove)

        design = np.hstack(hstack_list)
        print("Conditional Design Matrix Completed...")
        self.signal.print_signal.emit('Conditional Design Matrix Completed...')
        return design
    
    
    
    # Estimating model parameters
    def Estimated_location_scale(self, design, info_dict):
        
        """
        Return Matrix containing the following parts:
            - matrix of location, scale
            - matrix for variance
            - matrix for Regression coefficient estimates 
            ( group by center or group by event outcome or feature mean)
        """
        #Parameters are extracted from parameter dictionary
        n_batch = info_dict['n_batch']
        n_sample = info_dict['n_sample']
        sample_per_batch = info_dict['sample_per_batch']
        batch_info = info_dict['batch_info']
        mean_method = self.mean_method
        event_levels = self.event_levels
        X = self.dat

        betas = []
        betas_event = []
        
        #Least Squares Parameter Estimation
        for i in range(X.shape[0]):  
            betas.append(self.get_beta_with_nan(X[i,:], design))
        B_hat = np.vstack(betas).T

        #Estimate location and scale parameters by different grouping methods
        ### different sample centers  ###
        if mean_method == ['center']:
            print("The value of the feature is estimated according to different centers")
            self.signal.print_signal.emit('The value of the feature is estimated according to different centers')
            grand_mean = np.dot(design[:,:n_batch], B_hat[:n_batch,:]).T
            stand_mean  = grand_mean
        ### different sample event outcomes  ###
        elif mean_method == ['event']:
            print("The value of the feature is estimated according to different event outcomes")
            self.signal.print_signal.emit('The value of the feature is estimated according to different event outcomes')
            m_method = "event"
            event_design = self.make_design_matrix(m_method)
             #Least Squares Parameter Estimation
            for i in range(X.shape[0]):  
                betas_event.append(self.get_beta_with_nan(X[i,:], event_design))
            B_hat_event = np.vstack(betas_event).T              
            grand_mean = np.dot(event_design[:,:len(event_levels)], B_hat_event[:len(event_levels),:]).T  
            stand_mean  = grand_mean                 
        ### mean of feature centers ###
        else:
            print("The value of the feature is estimated by the mean of the feature")
            self.signal.print_signal.emit('The value of the feature is estimated by the mean of the feature')
            grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
            stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    
        ######### Continue here.     
        var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))
        
        print("Model parameter estimation complete...")
        self.signal.print_signal.emit('Model parameter estimation complete...')
        return grand_mean,stand_mean,var_pooled,B_hat,betas_event
    
    
    
    
    def standardize_across_features(self,design,info_dict,stand_mean,var_pooled,B_hat):  
        
        #Parameters are extracted from parameter dictionary
        reserve_cols = self.reserve_cols
        remove_cols  = self.remove_cols
        n_batch = info_dict['n_batch']
        n_sample = info_dict['n_sample']
        X = self.dat
        
        #Design the sample condition matrix to zero
        tmp = np.array(design.copy())
        tmp[:,:n_batch] = 0
        
        #Save relevant feature properties
        tmp_reserve = self.design_col_2_zero(remove_cols,n_batch,tmp,name_select = "remove")
        #Remove fixed influence factors
        tmp_remove = self.design_col_2_zero(reserve_cols,n_batch,tmp,name_select = "reserve")
     
        #Retain the correlated factor part and remove the fixed factor noise
        s_mean  = stand_mean + np.dot(tmp_reserve , B_hat).T
        s_mean  = stand_mean - np.dot(tmp_remove , B_hat).T
        
        #Feature sample normalization
        s_data = ((X - s_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))
        #
        print("Adjusting data across samples is done...")
        
        return s_data, s_mean
    

    #one-hot help function
    def to_categorical(self,y, nb_classes=None):
        
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
            
        return Y
               
    #Least Squares Parameter Estimation Function
    def get_beta_with_nan(self,yy, mod):
        
        wh = np.isfinite(yy)
        mod = mod[wh,:]
        yy = yy[wh]
        B = np.dot(np.dot(la.inv(np.dot(mod.T, mod)), mod.T), yy.T)
        
        return B

    #Design Matrix Adjustment Help Function    
    def design_col_2_zero(self,name_cols,n_batch,tmp_design_col,name_select):
        
        tmp_design_col_new = tmp_design_col.copy()
        if name_select == "reserve":                 
            for i in range(len(name_cols)):
                 col = i + n_batch 
                 tmp_design_col_new[:,col] = 0
        else:
            for i in range(len(self.reserve_cols),len(self.reserve_cols)+len(name_cols)):
                col = i + n_batch 
                tmp_design_col_new[:,col] = 0  
                
        return tmp_design_col_new
    
        
            



        
        
        
        
        
    
    
    
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



