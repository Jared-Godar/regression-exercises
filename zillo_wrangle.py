import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# function to contact database
def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

# function to query database and return zillow df
def get_zillo():
    query = """
    SELECT bedroomcnt as bedrooms, 
       bathroomcnt as bathrooms,
       calculatedfinishedsquarefeet as square_feet,
       taxvaluedollarcnt as home_value,
       yearbuilt as year,
	   taxamount as taxes,
       fips as fips_number
    FROM predictions_2017
    JOIN properties_2017 USING(id)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE #(transactiondate >= '2017-05-01' AND transactiondate <= '2017-06-30') 
        propertylandusetypeid = '261'
        AND bedroomcnt > 0
        AND bathroomcnt > 0
        AND calculatedfinishedsquarefeet > 0 
        AND taxamount > 0
        AND taxvaluedollarcnt > 0
        AND fips > 0
    ORDER BY fips;
    """
    df = pd.read_sql(query, get_db_url('zillow'))
    return df

# function to clean up my zillow df
def clean_data(df):
    '''
    This funciton takes in the zillow df and drops observations with Null values
    and handles data types returning a df with a basic clean.
    '''
    df = df.dropna()
    df['fips_number'] = df['fips_number'].astype(object)
    df['square_feet'] = df['square_feet'].astype(int)
    df['year'] = df['year'].astype(object)

    return df

#Outliers

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

# Split

def split_my_data(df):
    '''
    This function performs a 3-way split returning my train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)

    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test


#Scaler
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

# composite
def wrangle_zillow():
    df = get_zillo()
    df = clean_data(df)
    columns = ['bedrooms', 'bathrooms', 'square_feet', 'home_value', 'taxes']
    df = remove_outliers(df, 1.5, columns)
    train, validate, test = split_my_data(df)
    scaler = StandardScaler().fit(train[columns])
    scaler, train_scaled, validate_scaled, test_scaled = gen_scaler(columns, train, validate, test, scaler) 
    return scaler, train_scaled, validate_scaled, test_scaled