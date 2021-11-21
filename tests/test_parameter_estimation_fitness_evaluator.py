"""
This file will be used to test the parameter_estimation_fitness_evaluator.py file.
"""

# Import cleaning and splitting
from K_folds_data_splitter import K_folds_data_splitter
from SBstoat_model_fitting_to_folds import SBstoat_model_fitting_to_folds
from parameter_estimation_fitness_evaluator import parameter_estimation_fitness_evaluator
import tellurium as TE


# Importing libraries for property tests
import math
import pandas as pd


# In[2]:

data = pd.read_csv('data/test_function_data.csv', index_col=0)

df = K_folds_data_splitter(data, folds = 4)

antimony = """
    S1 -> S2; k1*S1;
    S2 -> S3; k2*S2;
    S3 -> S4; k3*S3

    S1 =10; S2 = 0; S3 = 0; S4 = 0;
    k1= 0;  k2 =0; k3=0
    """

rates = ['k1','k2','k3']

pathway_parameters = []
for rate in rates:   
    pathway_parameters.append(SB.Parameter(rate, lower=0, value=500, upper=10))
    
model = te.loada(antimony)

t_start = data.T.iloc[0].values[0]
t_end = data.T.iloc[0].values[-1]
t_steps = len(data.T.iloc[0].values)

sb_model_fit = SBstoat_model_fitting_to_folds(antimony, rates, df,pathway_parameters)

    

# In[3]:


def test_parameter_estimation_fitness_evaluator_1():
    '''
    Test to determine if number of rate estimation columns matches the input number of rate
    '''
    parameter_estimates = parameter_estimation_fitness_evaluator(sb_model_fit, df, folds, model)
    assert len(parameter_estimates.columns) == len(sb_model_fit.columns), (
        "R squared column was not added, try again")
    return




# In[4]:


def test_parameter_estimation_fitness_evaluator_2():
    '''
    Test to determine estimation was run over the number of folds provided
    '''
    parameter_estimates = parameter_estimation_fitness_evaluator(sb_model_fit, df, folds, model)
    for row in range(len(parameter_estimates)):
        assert isinstance(parameter_estimates['AIC'].iloc[row], float), (
            "AIC number is not a float! this will mess up down stream processing!")
    return



# In[5]:


def test_parameter_estimation_fitness_evaluator_3():
    '''
    Test to determine that the column names match rates list, important in downstream processing
    '''
    parameter_estimates = parameter_estimation_fitness_evaluator(sb_model_fit, df, folds, model)
    for row in range(len(parameter_estimates)):
        assert isinstance(parameter_estimates['BIC'].iloc[row], float), (
            "BIC number is not a float! this will mess up down stream processing!")
    return




# In[6]:


def test_parameter_estimation_fitness_evaluator_4():
    '''
    Test to determine that the column names match rates list, important in downstream processing
    '''
    parameter_estimates = parameter_estimation_fitness_evaluator(sb_model_fit, df, folds, model)
    for row in range(len(parameter_estimates)):
        assert isinstance(parameter_estimates['Reduced χ²'].iloc[row], float), (
            "Reduced χ² number is not a float! this will mess up down stream processing!")
    return




# In[7]:


def test_parameter_estimation_fitness_evaluator_5():
    '''
    Test to determine that the column names match rates list, important in downstream processing
    '''
    parameter_estimates = parameter_estimation_fitness_evaluator(sb_model_fit, df, folds, model)
    for row in range(len(parameter_estimates)):
        assert isinstance(parameter_estimates['χ²'].iloc[row], float), (
            "χ² number is not a float! this will mess up down stream processing!")
    return




# In[8]:


def test_parameter_estimation_fitness_evaluator_6():
    '''
    Test to determine that the column names match rates list, important in downstream processing
    '''
    parameter_estimates = parameter_estimation_fitness_evaluator(sb_model_fit, df, folds, model)
    for row in range(len(parameter_estimates)):
        assert isinstance(parameter_estimates['R²'].iloc[row], float), (
            "R² number is not a float! this will mess up down stream processing!")
    return