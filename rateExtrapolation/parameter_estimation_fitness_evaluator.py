''' This is the .py file that contains the parameter estimation fitness evaluator function'''

#imports
import numpy as np
import tellurium as te



def parameter_estimation_fitness_evaluator(data, estimates_df, cross_val_df, folds, rates, antimony):
    """ Simulates with K-fold approximations, returns Rsquared for each fold
    input
    -----
    dat, users input data
    estimates_df, returned by SBstoat_model_fitting_to_folds function
    cross_val_df, returned by K_folds_splitter
    folds, integer number of folds; default = 25
    rates, user input, list of strings of rates
    model: TE roadrunner

    output
    ------
    estimates_df with added R squared column"""

    #determine start and end of data such that it can be called to later
    t_start = data.T.iloc[0].values[0]
    t_end = data.T.iloc[0].values[-1]
    t_steps = len(data.T.iloc[0].values)

    #load model using antimony string
    model = te.loada(antimony)

    r_squared = [] #empty to append to

    #calculate R squared and residuals for all folds
    for fold in range(folds):

        #re-initialize each rate and simulate with start and stop from exp data
        for rate in rates:
            model.reset()
            model[rate] = estimates_df[rate][fold]
        simulated_ = model.simulate(t_start, t_end, t_steps)


        inner_list_res = [] #catch list for residuals that occur at each timepoint
        inner_list_act = []

        #Calculate residuals by finding closest value in the test datasets for each fold
        for row in range(len(cross_val_df[fold]['Test']['time'])):
            row_val = cross_val_df[fold]['Test']['time'].iloc[row]

            #finding the closest index to the test value
            a_list = list(abs(simulated_['time'] - row_val))
            min_value = min(a_list)
            index_of_best_fit =  a_list.index(min_value)

            #calculate the residuals for each column from simualted and test values
            for column in range(len(simulated_.T)):
                if column != 0:
                    predicted_value = simulated_.T[column][index_of_best_fit]
                    actual_value = cross_val_df[fold]['Test'].T.iloc[column].iloc[row]
                    inner_list_res.append(predicted_value - actual_value)
                    inner_list_act.append(cross_val_df[fold]['Test'].T.iloc[column].iloc[row])

        r_squared.append(1 - np.var(inner_list_res)/np.var(inner_list_act))

    estimates_df['R²'] = r_squared

    estimates_df = estimates_df.sort_values(by = 'AIC', ascending = True)
    return estimates_df
