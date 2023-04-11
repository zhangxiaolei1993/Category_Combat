# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:39:35 2022

@author: dell
"""


from __future__ import absolute_import, print_function
import pandas as pd
import numpy as np
import numpy.linalg as la
import math

class Bayesian_process():
    
    def __init__(self,covars,parameter_dict,dat, signal):
        #parameter initialization
        self.signal = signal
        self.covars = covars
        self.dat = np.array(dat, dtype='float32')
        self.batch_col = parameter_dict['batch_col']
        self.reserve_cols = parameter_dict['reserve_cols']
        self.remove_cols = parameter_dict['remove_cols'] 
        self.event_col = parameter_dict['event_col']
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
        #     else:
        #         self.reserve_cols = [reserve_cols]
           
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

                
 
    def fit_LS_model_and_find_priors(self,s_data, design, info_dict):
        
        #Parameters are extracted from parameter dictionary
        n_batch = info_dict['n_batch']
        batch_info = info_dict['batch_info'] 
        mean_only= self.mean_only
        batch_design = design[:,:n_batch]
        
        #Normalized data mean parameter estimates
        gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

        #Standardized data variance estimate
        delta_hat = []
        for i, batch_idxs in enumerate(batch_info):
            if mean_only:   
                delta_hat.append(np.repeat(1, s_data.shape[0]))
            else:
                delta_hat.append(np.var(s_data[:,batch_idxs],axis=1,ddof=1))
                
        #Distribution parameter estimates for the mean of the normalized data
        gamma_bar = np.mean(gamma_hat, axis=1) 
        t2 = np.var(gamma_hat,axis=1, ddof=1)
        
        #Distribution parameter estimates of the variance of the standardized data
        if mean_only:
            a_prior = None
            b_prior = None
        else:
            a_prior = list(map(self.aprior, delta_hat))
            b_prior = list(map(self.bprior, delta_hat))
        
        #Returns a dictionary of prior distributions
        LS_dict = {}
        LS_dict['gamma_hat'] = gamma_hat
        LS_dict['delta_hat'] = delta_hat
        LS_dict['gamma_bar'] = gamma_bar
        LS_dict['t2'] = t2
        LS_dict['a_prior'] = a_prior
        LS_dict['b_prior'] = b_prior
        print("Bayesian prior distribution parameter estimation has been completed...")
        self.signal.print_signal.emit('Bayesian prior distribution parameter estimation has been completed...')
        return LS_dict
        
   
    
    def find_parametric_adjustments(self,s_data, LS, info_dict):
    
        #Parameters are extracted from parameter dictionary
        batch_info  = info_dict['batch_info'] 
        mean_only= self.mean_only
        
        #Posterior Distribution Parameter Estimation
        gamma_star, delta_star = [], []
        
        for i, batch_idxs in enumerate(batch_info):
            if mean_only:
                gamma_star.append(self.postmean(LS['gamma_hat'][i], LS['gamma_bar'][i], 1, 1, LS['t2'][i]))
                delta_star.append(np.repeat(1, s_data.shape[0]))               
            else:
                temp = self.it_sol(s_data[:,batch_idxs], LS['gamma_hat'][i],
                            LS['delta_hat'][i], LS['gamma_bar'][i], LS['t2'][i], 
                            LS['a_prior'][i], LS['b_prior'][i])
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
    
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)
        print("Bayesian Posterior Distribution Parameter Estimation Completed...")
        self.signal.print_signal.emit('Bayesian Posterior Distribution Parameter Estimation Completed...')
        return gamma_star, delta_star
    
    
    def find_non_parametric_adjustments(self,s_data, LS, info_dict):
        
        #Parameters are extracted from parameter dictionary
        batch_info  = info_dict['batch_info'] 
        mean_only= self.mean_only
        
        #Posterior Distribution Parameter Estimation
        gamma_star, delta_star = [], []
        for i, batch_idxs in enumerate(batch_info):
            if mean_only:
                LS['delta_hat'][i] = np.repeat(1, s_data.shape[0])   
                
            temp = self.int_eprior(s_data[:,batch_idxs], LS['gamma_hat'][i],
                        LS['delta_hat'][i])    
            gamma_star.append(temp[0])
            delta_star.append(temp[1])
    
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)
  
        return gamma_star, delta_star
    
    
    def find_non_eb_adjustments(self,s_data, LS, info_dict):
        
        #Parameters are extracted from parameter dictionary
        gamma_star = np.array(LS['gamma_hat'])
        delta_star = np.array(LS['delta_hat'])
    
        return gamma_star, delta_star
    
    
    def adjust_data_final(self,s_data, design, gamma_star, delta_star, s_mean, var_pooled, info_dict):

        #Parameters are extracted from parameter dictionary
        d_c   =   self.discretization_coefficient
        print("The coefficient of Category_coefficient are",d_c)
        self.signal.print_signal.emit(f'The coefficient of Category_coefficient are {d_c}')
        sample_per_batch = info_dict['sample_per_batch']
        n_batch = info_dict['n_batch']
        n_sample = info_dict['n_sample']
        batch_info = info_dict['batch_info']
        
        #Sample Condition Matrix Extraction
        batch_design = design[:,:n_batch]
    
        bayesdata = s_data
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)
    
        for j, batch_idxs in enumerate(batch_info):
            dsq = np.sqrt(delta_star[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom = np.dot(dsq, np.ones((1, sample_per_batch[j])))
            numer = np.array(bayesdata[:,batch_idxs] - np.dot(batch_design[batch_idxs,:], gamma_star).T)
            bayesdata[:,batch_idxs] = numer / denom
        
        #The relevant parameters are brought into the final adjustment formula
        vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
        bayesdata = d_c * bayesdata * np.dot(vpsq, np.ones((1, n_sample))) + s_mean
        print("Final data adjustment ...")
        self.signal.print_signal.emit(f'Final data adjustment ...')
        return bayesdata
    
    
    

    def aprior(self,delta_hat):
        m = np.mean(delta_hat)
        s2 = np.var(delta_hat,ddof=1)
        
        return (2 * s2 +m**2) / float(s2)

    def bprior(self,delta_hat):
        m = delta_hat.mean()
        s2 = np.var(delta_hat,ddof=1)
        
        return (m*s2+m**3)/s2
    
    def postmean(self,g_hat, g_bar, n, d_star, t2):
        
        return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)
    
    def postvar(self,sum2, n, a, b):
        
        return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
    
    #Helper function for parametric adjustements:
    def it_sol(self,sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
        n = (1 - np.isnan(sdat)).sum(axis=1)
        g_old = g_hat.copy()
        d_old = d_hat.copy()
    
        change = 1
        count = 0
        while change > conv:
            g_new = self.postmean(g_hat, g_bar, n, d_old, t2)
            sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
            d_new = self.postvar(sum2, n, a, b)
    
            change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
            g_old = g_new #.copy()
            d_old = d_new #.copy()
            count = count + 1
        adjust = (g_new, d_new)
        
        return adjust 
    
      
    #Helper function for non-parametric adjustements:
    def int_eprior(self,sdat, g_hat, d_hat):
        
        r = sdat.shape[0]
        gamma_star, delta_star = [], []
        for i in range(0,r,1):
            g = np.delete(g_hat,i)
            d = np.delete(d_hat,i)
            x = sdat[i,:]
            n = x.shape[0]
            j = np.repeat(1,n)
            A = np.repeat(x, g.shape[0])
            A = A.reshape(n,g.shape[0])
            A = np.transpose(A)
            B = np.repeat(g, n)
            B = B.reshape(g.shape[0],n)
            resid2 = np.square(A-B)
            sum2 = resid2.dot(j)
            LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
            LH = np.nan_to_num(LH)
            gamma_star.append(sum(g*LH)/sum(LH))
            delta_star.append(sum(d*LH)/sum(LH))
        adjust = (gamma_star, delta_star)
        
        return adjust

    


































        