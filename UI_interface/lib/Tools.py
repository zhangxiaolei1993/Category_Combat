# -*- coding: UTF-8 -*-
'''
@Time    : 2023/4/11 11:54
@Author  : 魏林栋
@Site    : 
@File    : Tools.py
@Software: PyCharm
'''
import os
import numpy.linalg as la

# os.chdir(r"E:\keyan\github\Category_Combat\Category_Combat")
from lib.standardize_across_features import Location_scale_model
from lib.bayesain_procession import Bayesian_process


def aa():
    print('hello')


def Category_Combat(covars, parameter_dict, dat, signal):
    ############################################Location scale model
    # Location scale model initialization
    L_S_model = Location_scale_model(covars, parameter_dict, dat, signal)

    # Location-scale model data preprocessing
    batch_levels, sample_per_batch, event_levels, sample_per_event, info_dict = L_S_model.data_procession()

    # Design Matrix
    design = L_S_model.make_design_matrix(None)

    # Location-scale model parameter estimation
    grand_mean, stand_mean, var_pooled, B_hat, betas_event \
        = L_S_model.Estimated_location_scale(design, info_dict)

    # Location-scale model normalization
    s_data, s_mean = L_S_model.standardize_across_features(
        design, info_dict, stand_mean, var_pooled, B_hat)

    ###########################################Bayesian process
    # Bayesian model initialization
    L_S_bayesian = Bayesian_process(covars, parameter_dict, dat, signal)

    # Bayesian prior distribution parameter estimation
    LS_dict = L_S_bayesian.fit_LS_model_and_find_priors(
        s_data, design, info_dict)

    # Empirical Bayesian Posterior Distribution Parameter Estimation
    gamma_star, delta_star = L_S_bayesian.find_parametric_adjustments(s_data,
                                                                      LS_dict, info_dict)

    # Non-empirical Bayesian Posterior Distribution Parameter Estimation
    # gamma_star, delta_star = L_S_bayesian.find_non_parametric_adjustments(s_data,
    #                         LS_dict, info_dict)

    # Bayesian adjustment process
    bayesdata = L_S_bayesian.adjust_data_final(s_data, design, gamma_star, delta_star,
                                               s_mean, var_pooled, info_dict)

    return bayesdata