import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from env import host, user, password

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

# composite
def wrangle_zillow():
    df = get_zillo()
    df = clean_data(df)
    return df 