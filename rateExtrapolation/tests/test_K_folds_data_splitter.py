"""
This file will be used to test the K_folds_data_splitter.py file.
"""

# Import cleaning and splitting
from K_folds_data_splitter import K_folds_data_splitter

# Importing libraries for property tests
import math
import pandas as pd
import numpy as np


# In[2]:

data = pd.read_csv('data/test_function_data.csv')
data.pop('Unnamed: 0')

# In[3]:


def test_k_folds_data_splitter_1():
    '''
    Test to determine if number of columns matches the input number of folds
    '''
    folds = 4
    df = K_folds_data_splitter(data, folds)
    folds_num = len(df.columns)
    assert folds_num == folds, (
        "Number of columns in consistent with number of folds, there may be an assignment error")

    return


# In[4]:


def test_k_folds_data_splitter_2():
    '''
    Test to determine if rows for test and train data are returned
    '''
    folds = 4
    df = K_folds_data_splitter(data,folds)
    rows_num = len(df)
    assert rows_num == 2, (
        "Number of rows is inconsistent with Test/Train, function assignment has failed")
    return


# In[5]:


def test_k_folds_data_splitter_3():
    '''
    Test to determine if number of samples in test set is less than sample number of train set
    '''
    folds = 4
    df = K_folds_data_splitter(data,folds)
    test_set_length = df.loc['Test'][0]
    train_set_length = df.loc['Train'][0]
    
    assert test_set_length < train_set_length, (
        "Test set contains more samples than train_set, issues with assignment, may need to decrease fold number")
    return

