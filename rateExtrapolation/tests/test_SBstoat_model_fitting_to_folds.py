"""
This file will be used to test the SBstoat_model_fitting_to_folds.py file.
"""

# Import cleaning and splitting
from K_folds_data_splitter import K_folds_data_splitter
from SBstoat_model_fitting_to_folds import SBstoat_model_fitting_to_folds

# Importing libraries for property tests
import math
import pandas as pd
import numpy as np
import SBstoat as SB


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

# In[3]:


def test_SBstoat_model_fitting_to_folds_1():
    '''
    Test to determine if number of rate estimation columns matches the input number of rate
    '''
    sb_model_fit = SBstoat_model_fitting_to_folds(antimony, rates, df,pathway_parameters)
    assert (len(sb_model_fit.columns)-4) == len(rates), (
        "Number of estimated rates does not equal number of designated rates")
    return


# In[4]:


def test_SBstoat_model_fitting_to_folds_2():
    '''
    Test to determine estimation was run over the number of folds provided
    '''
    sb_model_fit = SBstoat_model_fitting_to_folds(antimony, rates, df,pathway_parameters)
    assert len(sb_model_fit) == len(df.T), (
        "Number of fold estimations does not equal number of folds given")
    return


# In[5]:


def test_SBstoat_model_fitting_to_folds_3():
    '''
    Test to determine that the column names match rates list, important in downstream processing
    '''
    sb_model_fit = SBstoat_model_fitting_to_folds(antimony, rates, df,pathway_parameters)

    assert list(sb_model_fit.columns[0:len(rates)]) == rates, (
        "rates returned do not match rates provided, check case and antimony string")
    return
