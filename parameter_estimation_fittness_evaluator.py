
import math
import numpy as np
import pandas as pd

import tellurium as te



def parameter_estimation_fittness_evaluator(estimates_df, folds, model):
    """ Simulates with K-fold approximations, returns Rsquared for each fold
    input
    -----
    estimates_df, returned by SBstoat_model_fitting_to_folds function
    folds: integer number of folds; default = 25
    model: TE roadrunner
    
    output
    ------
    estimates_df with added R squared column"""
    r_squared = []

    for fold in range(folds):

        #re-initialize each rate and simulate with start and stop from exp data
        for rate in rates:
            model.reset()
            model[rate] = estimates_df[rate][fold]
        s = model.simulate(t_start, t_end, t_steps)


        inner_list_res = [] #catch list for residuals that occur at each timepoint
        inner_list_act = []

        #Calculate residuals by finding closest value in the test datasets for each fold
        for row in range(len(cross_val_df[fold]['Test']['time'])):

            row_val = cross_val_df[fold]['Test']['time'].iloc[row]
            #finding the closest index to the test value
            a_list = list(abs(s['time'] - row_val))
            min_value = min(a_list)
            index_of_best_fit =  a_list.index(min_value)

            #calculate the residuals for each column from simualted and test values
            for column in range(len(s.T)):
                if column != 0:
                    name = cross_val_df.columns[column]
                    inner_list_res.append(s.T[column][index_of_best_fit] - cross_val_df[fold]['Test'].T.iloc[column].iloc[row])
                    inner_list_act.append(cross_val_df[fold]['Test'].T.iloc[column].iloc[row])

        r_squared.append(1 - np.var(inner_list_res)/np.var(inner_list_act))  

    estimates_df['R²'] = r_squared

    estimates_df = estimates_df.sort_values(by = 'AIC', ascending = True)
    return estimates_df