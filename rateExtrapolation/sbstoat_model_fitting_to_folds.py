'''this .py file contains the function sbstoat_model_fitting_to_folds function'''

import numpy as np
import pandas as pd
from SBstoat.namedTimeseries import NamedTimeseries
from SBstoat.modelFitter import ModelFitter




def sbstoat_model_fitting_to_folds(antimony, rates, cross_val_df,pathway_parameters):
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

        parameter_estimates=dict(fitter.params.valuesdict())
        temp_dict = dict()
        for rate in rates:
            temp_dict[rate] = parameter_estimates[rate]




        #chi-squared value
        chi_sqr_start= (fitter.reportFit().find('chi-square         = ')+
            len('chi-square         = '))
        chi_sqr_end = fitter.reportFit()[chi_sqr_start:].find('\n') + chi_sqr_start
        chi_sqr = fitter.reportFit()[chi_sqr_start:chi_sqr_end]

        temp_dict['χ²'] = np.round(float(chi_sqr),2)



        #reduced chi squared
        chi_sqr_rd_start = (fitter.reportFit().find('reduced chi-square = ')+
            len('reduced chi-square = '))
        chi_sqr_rd_end = fitter.reportFit()[chi_sqr_rd_start:].find('\n') + chi_sqr_rd_start
        chi_sqr_rd = fitter.reportFit()[chi_sqr_rd_start:chi_sqr_rd_end]

        temp_dict['Reduced χ²'] = np.round(float(chi_sqr_rd),2)



        #Akaike info crit
        aic_start = (fitter.reportFit().find('Akaike info crit   = ')+
            len('Akaike info crit   = '))
        aic_end = fitter.reportFit()[aic_start:].find('\n') + aic_start
        aic = fitter.reportFit()[aic_start:aic_end]

        temp_dict['AIC'] = np.round(float(aic),2)

        #Bayesian info crit
        bic_start = (fitter.reportFit().find('Bayesian info crit = ')+
            len('Bayesian info crit = '))
        bic_end = fitter.reportFit()[bic_start:].find('\n') + bic_start
        bic = fitter.reportFit()[bic_start:bic_end]

        temp_dict['BIC'] = np.round(float(bic),2)





        estimates_df = estimates_df.append(temp_dict, ignore_index=True)
    return estimates_df
