#Package Import
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import datetime
import tellurium as te
import SBstoat as SB
import urllib.request
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from SBstoat.modelFitter import ModelFitter





def K_folds_data_splitter(experimental_data, folds):
    """ Splits data into k-folds train and validation sets.
    
    Inputs
    ------
    experimental_data: DataFrame
    folds: number of folds (integer)
        - Default: 25
        
    Outputs
    -------
    cross_val_df
        - Dataframe of K folds columns x 2 rows (Test, Train)
        - each cell is a dataframe of values
    
    """
    cross_val_df = pd.DataFrame()
    indices = np.arange(len(experimental_data))

    for fold_num in range(folds):
        test_set = []
        train_set = []
        test_set = experimental_data.iloc[[n for n in indices if n % folds == fold_num],:]
        train_set = experimental_data.iloc[[n for n in indices if n % folds != fold_num],:]
        cross_val_df[fold_num] = [test_set,train_set]
    cross_val_df = cross_val_df.rename(index = {0:'Test', 1:'Train'})
    return cross_val_df




def SBstoat_model_fitting_to_folds(cross_val_df,pathway_parameters):
''' Estimations for parameters listed in pathway parameters over number of folds.

INPUTS
------
cross-validation_df: DataFrame (2 rows (Train/Test) x K-folds columns)
pathway-parameters: list of SBStoat parameter objects


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
    fitter = ModelFitter(antimony, 
                         NamedTimeseries(dataframe=cross_val_df[fold]['Train']), 
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


    chi_sqr_start = fitter.reportFit().find('chi-square         = ')+ len('chi-square         = ')
    chi_sqr_end = fitter.reportFit()[chi_sqr_start:].find('\n') + chi_sqr_start
    chi_sqr = fitter.reportFit()[chi_sqr_start:chi_sqr_end]

    temp_dict['χ²'] = np.round(float(chi_sqr),2)


    chi_sqr_reduced_start = fitter.reportFit().find('reduced chi-square = ')+ len('reduced chi-square = ')
    chi_sqr_reduced_end = fitter.reportFit()[chi_sqr_reduced_start:].find('\n') + chi_sqr_reduced_start
    chi_sqr_reduced = fitter.reportFit()[chi_sqr_reduced_start:chi_sqr_reduced_end]

    temp_dict['Reduced χ²'] = np.round(float(chi_sqr_reduced),2)


    AIC_start = fitter.reportFit().find('Akaike info crit   = ')+ len('Akaike info crit   = ')
    AIC_end = fitter.reportFit()[AIC_start:].find('\n') + AIC_start
    AIC = fitter.reportFit()[AIC_start:AIC_end]

    temp_dict['AIC'] = np.round(float(AIC),2)


    BIC_start = fitter.reportFit().find('Bayesian info crit = ')+ len('Bayesian info crit = ')
    BIC_end = fitter.reportFit()[BIC_start:].find('\n') + BIC_start
    BIC = fitter.reportFit()[BIC_start:BIC_end]

    temp_dict['BIC'] = np.round(float(BIC),2)



    estimates_df = estimates_df.append(temp_dict, ignore_index=True)
    estimates_df
return estimates_df




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






def rate_Extrapolation(experimental_data, antimony, rates, folds = 25):
    '''takes in data, model, rates, and fold and determines the best fit rate constants
    
    input
    -----
    experimental_data: either csv or dataframe where the first column is the time and subsequent columns are substrate concentrations
    antimony: tellurium model type(String)
    rates_to_estimate: list of strings of rates ex. ['k1','k2'] 
    folds: integer to control number of folds. default = 25
    
    outputs
    -------
    simulated_data, dataframe tellurium model simulation
    three figures: 
        - simulated and modeled plots + residual plots for each substrate
        - histograms of rates extrapolated and plots to show whether rates are correlated
        - final plot of model with statistics of the model
    pdf of three output plots saved to path
    excel sheet of model simulated data'''
    
    # determining time span to simulate TE road runner
    # designates in the inital and final value of the time column...
    # and the number of instances to determin [start,step,end]
    t_start = experimental_data.T.iloc[0].values[0]
    t_end = experimental_data.T.iloc[0].values[-1]
    t_steps = len(experimental_data.T.iloc[0].values)

    # Setting up list of pathway parameters
    pathway_parameters = []
    for rate in rates:   
        pathway_parameters.append(SB.Parameter(rate, lower=0, value=500, upper=10))


    #generates Road-Runner Model for downstream analysis
    model = te.loada(antimony)
    
    cross_val_df = K_folds_data_splitter(experimental_data, folds)
    print('Data Split!')
    print('Parameters Estimating...')
    estimates_df = SBstoat_model_fitting_to_folds(cross_val_df,pathway_parameters)
    print('Parameters Estimated!')
    estimates_df = parameter_estimation_fittness_evaluator(estimates_df, folds,model)
    print('Estimations evaluated!')
    
    #Final Model Simulation with chosen best rates
    for rate in rates:
        model.reset()
        model[rate] = estimates_df[rate].iloc[0]
    simulated = model.simulate(t_start, t_end, t_steps)
    
    %matplotlib inline
    #first figure
    x_data = experimental_data.T.iloc[0].T
    y_data = experimental_data.T.iloc[1:].T
    fig1,axes = plt.subplots(ncols=2, nrows= len(y_data.T), figsize = [11,6*len(y_data.T)])
    y_cols = y_data.columns
    for i in range(len(y_data.T)):
        name = y_cols[i]
        residuals = y_data[name]-simulated.T[i+1]
        text_str = 'µ = ' + str(np.mean(residuals))
        sub_name_exp = 'S'+str(i+1)+' Experimental'
        sub_name_modeled = 'S'+str(i+1)+' Modeled'
        axes[i,0].scatter(x_data,y_data[name],s=20, facecolors='none', edgecolors='b',label = sub_name_exp)
        axes[i,0].plot(x_data,simulated.T[i+1],label = sub_name_modeled, color = 'red')
        axes[i,0].set_ylim([0,pd.DataFrame(y_data).max().max() + 0.25 * pd.DataFrame(y_data).max().max()])
        axes[i,0].set_xlabel('Time')
        axes[i,0].set_ylabel('Concentration')
        axes[i,0].set_title('S'+str(i+1)+' Modeled vs Experimental', fontweight = 'bold')
        axes[i,0].legend()
        axes[i,1].scatter(x_data,residuals, alpha = 0.5, s=45)
        axes[i,1].set_ylim([residuals.min()+0.75*residuals.min(),residuals.max()+0.75*residuals.max()])
        axes[i,1].plot(x_data,np.zeros(len(x_data)), 'k-')
        axes[i,1].set_xlabel('Time')
        axes[i,1].set_ylabel('Residuals')
        axes[i,1].set_title('S'+str(i+1)+' Residuals Plot', fontweight = 'bold')
        props = dict( facecolor='white', alpha=0.5)
        axes[i,1].text(0.025, 0.075, text_str, transform=axes[i,1].transAxes, fontsize=12, verticalalignment='top', bbox=props)

    
    #second figure
    fig2,axes = plt.subplots(ncols= len(rates), nrows= len(rates), figsize = [6*len(rates),6*len(rates)])
    y_cols = y_data.columns
    for j in range(len(rates)):
        name_1 = rates[j]
        for i in range(len(rates)):
            name = rates[i]
            if i == j:
                axes[i,j].hist(list(estimates_df[name].values), bins = folds*2)
                axes[i,j].set_xlabel('Extrapolated Rates')
                axes[i,j].set_ylabel('Count')
                axes[i,j].set_title(name+' Exptrapolated Rates', fontweight = 'bold')
                props = dict( facecolor='white', alpha=0.5)
                txt_str = 'Best ' + name+' = '+ str(np.round(estimates_df[name][0],3))
                axes[i,j].text(0.65, 0.95, txt_str, transform=axes[i,j].transAxes, fontsize=12, verticalalignment='top', bbox=props)
            elif i < j:
                axes[i,j].axis('off')
            else:
                axes[i,j].scatter(estimates_df[name].values, estimates_df[name_1].values, s = 45, alpha = 0.5)
                axes[i,j].set_xlabel(name)
                axes[i,j].set_ylabel(name_1)
                axes[i,j].set_xlim(estimates_df[rates].values.min()-0.25*abs(estimates_df[rates].values.min()), estimates_df[rates].values.max()+ 0.25*abs(estimates_df[rates].values.max()))
                axes[i,j].set_ylim(estimates_df[rates].values.min()-0.25*abs(estimates_df[rates].values.min()), estimates_df[rates].values.max()+ 0.25*abs(estimates_df[rates].values.max()))
                axes[i,j].set_title(name_1+' vs. '+name, fontweight = 'bold')
                
                
    #third figure
    columns = list(estimates_df.columns)
    rows = ['Best']
    fig3,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,7))
    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.15, len(columns)))

    # Plot bars and create text labels for the table
    cell_text = [[e] for e in list(np.round(estimates_df.iloc[0].astype(float),3).astype(str))]

    for i in range(len(simulated.T[1:])):
        sub_name_modeled = 'S'+str(i+1)+' Modeled'
        ax[0].plot(x_data,simulated.T[i+1],label = sub_name_modeled, linewidth = 2)

    ax[0].set_ylabel("Concentration", fontsize = 14)
    ax[0].set_xlabel("Time", fontsize = 14)

    ax[0].set_title('Modeled Reaction', fontsize = 14, fontweight = 'bold')
    ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize = 12)


    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowColours=colors,
                          rowLabels=columns,
                          loc='lower left')

    the_table.set_fontsize(12)
    the_table.scale(0.5, 1.75)
    ax[1].axis('off')

    #generating output dataframe
    output_dataframe = pd.DataFrame()
    for I in range(len(simulated.T)):
        sub_name_exp = 'S'+str(I)
        if I == 0:
            output_dataframe['time'] = simulated.T[I]
        else:
            output_dataframe[sub_name_exp] = simulated.T[I]
    
    
    #exporting data to 
    outputstring_pdf = str(datetime.date.today())+'_rateExtrapolation_plots.pdf'
    outputstring_excel = str(datetime.date.today())+'_rateExtrapolation_model_data.xlsx'
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputstring_pdf)
    figures = [fig1, fig2, fig3]
    for fig in figures:
        pdf.savefig( fig )
    pdf.close()

    output_dataframe.to_excel(outputstring_excel)
    
    print('PDF saved in local directory as : ' + outputstring_pdf)
    print('Excel sheet of simulations saved in local directory as : ' + outputstring_excel)
    
    return output_dataframe

