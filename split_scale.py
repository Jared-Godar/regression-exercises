import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_my_data(df):
    '''
    This function performs a 3-way split returning my train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)

    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test


def standard_scaler(columns_to_scale, train, validate, test):
    """
    Takes in train, validate, and test dfs with numeric values only
    Returns scaler, train_scaled, validate_scaled, test_scaled dfs
    Using a StandardScaler.
    """
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    scaler = StandardScaler().fit(train[columns_to_scale])
    
    train_scaled = pd.concat([
                        train,
                        pd.DataFrame(scaler.transform(train[columns_to_scale]), 
                        columns=new_column_names, 
                        index=train.index)],
                        axis=1)
    
    validate_scaled = pd.concat([
                        validate,
                        pd.DataFrame(scaler.transform(validate[columns_to_scale]), 
                        columns=new_column_names, 
                        index=validate.index)],
                        axis=1)
    
    test_scaled = pd.concat([
                        test,
                        pd.DataFrame(scaler.transform(test[columns_to_scale]), 
                        columns=new_column_names, 
                        index=test.index)],
                        axis=1)
    
    return scaler, train_scaled, validate_scaled, test_scaled


def gen_scaler(columns_to_scale, train, validate, test, scaler):
    """
    Takes in a a list of string names for columns, train, validate, 
    and test dfs with numeric values only, and a scaler and 
    returns scaler, train_scaled, validate_scaled, test_scaled dfs
    """
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    scaler.fit(train[columns_to_scale])
    
    train_scaled = pd.concat([
                        train,
                        pd.DataFrame(scaler.transform(train[columns_to_scale]), 
                        columns=new_column_names, 
                        index=train.index)],
                        axis=1)
    
    validate_scaled = pd.concat([
                        validate,
                        pd.DataFrame(scaler.transform(validate[columns_to_scale]), 
                        columns=new_column_names, 
                        index=validate.index)],
                        axis=1)
    
    test_scaled = pd.concat([
                        test,
                        pd.DataFrame(scaler.transform(test[columns_to_scale]), 
                        columns=new_column_names, 
                        index=test.index)],
                        axis=1)
    
    return scaler, train_scaled, validate_scaled, test_scaled