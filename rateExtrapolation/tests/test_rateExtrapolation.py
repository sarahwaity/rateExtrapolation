"""
This file will be used to test the rateExtrapolation.py file.
"""

# Import cleaning and splitting

from rate_Extrapolation import rateExtrapolation

# Importing libraries for property tests
import math
import pandas as pd


# In[2]:

data = pd.read_csv('data/test_function_data.csv', index_col=0)

antimony = """
    S1 -> S2; k1*S1;
    S2 -> S3; k2*S2;
    S3 -> S4; k3*S3

    S1 =10; S2 = 0; S3 = 0; S4 = 0;
    k1= 0;  k2 =0; k3=0
    """

rates = ['k1','k2','k3']

    

# In[3]:


def test_rateExtrapolation_1():
    '''
    Test to determine if modeled data has the same columns as input experimental data
    '''
    try_it_out,f1,f2,f3 = rateExtrapolation(data, antimony, rates, folds = 25)
    assert len(try_it_out.columns) == len(data.columns), (
        "Output data does not have same number of substrates as input data")
    return




# In[4]:


def test_rateExtrapolation_2():
    '''
    Test to determine if modeled data has the same number of rows as input experimental data
    '''
    try_it_out,f1,f2,f3 = rateExtrapolation(data, antimony, rates, folds = 25)
    assert len(try_it_out) == len(data), (
        "Output data does not have same number of substrates as input data")
    return
