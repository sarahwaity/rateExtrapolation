'''This .py file contains the function k_folds_data_splitter described below'''
import pandas as pd
import numpy as np


def k_folds_data_splitter(experimental_data, folds):
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
