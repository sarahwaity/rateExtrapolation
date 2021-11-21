
import numpy as np
import pandas as pd
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from SBstoat.modelFitter import ModelFitter




def SBstoat_model_fitting_to_folds(antimony, rates, cross_val_df,pathway_parameters):
    ''' Estimations for parameters listed in pathway parameters over number of folds.
    
    INPUTS
    ------
    cross-validation_df: DataFrame (2 rows (Train/Test) x K-folds columns)
    pathway-parameters: list of SBStoat parameter objects
    rates: list of strings that correspond to rates
    antimony: tellurium string of model 
    
    
    OUTPUTS
    -------
    estimates_df (DataFrame)
        - one column for each parameter estimated
        - one column for important stats from each fit
            - fitting method
            - chi squared
            - reduced chi squared
            - Akaike Info Crit
            - Bayesian Info Crit
        
    '''
    estimates_df = pd.DataFrame(columns = rates)
    
    for fold in range(len(cross_val_df.columns)):
        col_name = cross_val_df.columns[fold]
        fitter = ModelFitter(antimony, 
                             NamedTimeseries(dataframe=cross_val_df[col_name]['Train']), 
                             parametersToFit=pathway_parameters)
        fitter.fitModel()
        internal_meta_data = []
        fitter_output = fitter.reportFit()
        parameter_estimates=dict(fitter.params.valuesdict())
        temp_dict = dict()
        for rate in rates:
            temp_dict[rate] = parameter_estimates[rate]

        fitting_method_start = fitter.reportFit().find('# fitting method   = ')+ len('# fitting method   = ')
        fitting_method_end = fitter.reportFit()[fitting_method_start:].find('\n') + fitting_method_start
        fitting_method = fitter.reportFit()[fitting_method_start:fitting_method_end]

        #temp_dict['Fitting Method'] = fitting_method

        # print(fitting_method_start)
        # print(fitting_method_end)


        chi_sqr_start = fitter.reportFit().find('chi-square         = ')+ len('chi-square         = ')
        chi_sqr_end = fitter.reportFit()[chi_sqr_start:].find('\n') + chi_sqr_start
        chi_sqr = fitter.reportFit()[chi_sqr_start:chi_sqr_end]

        temp_dict['χ²'] = np.round(float(chi_sqr),2)

        # print(chi_sqr_start)
        # print(chi_sqr_end)
        # print(fitter.reportFit()[chi_sqr_start:chi_sqr_end])

        chi_sqr_reduced_start = fitter.reportFit().find('reduced chi-square = ')+ len('reduced chi-square = ')
        chi_sqr_reduced_end = fitter.reportFit()[chi_sqr_reduced_start:].find('\n') + chi_sqr_reduced_start
        chi_sqr_reduced = fitter.reportFit()[chi_sqr_reduced_start:chi_sqr_reduced_end]

        temp_dict['Reduced χ²'] = np.round(float(chi_sqr_reduced),2)

        # print(chi_sqr_reduced_start)
        # print(chi_sqr_reduced_end)
        # print(fitter.reportFit()[chi_sqr_reduced_start:chi_sqr_reduced_end])

        AIC_start = fitter.reportFit().find('Akaike info crit   = ')+ len('Akaike info crit   = ')
        AIC_end = fitter.reportFit()[AIC_start:].find('\n') + AIC_start
        AIC = fitter.reportFit()[AIC_start:AIC_end]

        temp_dict['AIC'] = np.round(float(AIC),2)

        # print(AIC_start)
        # print(AIC_end)
        # print(fitter.reportFit()[AIC_start:AIC_end])


        BIC_start = fitter.reportFit().find('Bayesian info crit = ')+ len('Bayesian info crit = ')
        BIC_end = fitter.reportFit()[BIC_start:].find('\n') + BIC_start
        BIC = fitter.reportFit()[BIC_start:BIC_end]

        temp_dict['BIC'] = np.round(float(BIC),2)

        # print(BIC_start)
        # print(BIC_end)
        #print(fitter.reportFit()[BIC_start:BIC_end])



        estimates_df = estimates_df.append(temp_dict, ignore_index=True)
        estimates_df
    return estimates_df
