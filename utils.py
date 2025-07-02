import pandas as pd
import numpy as np


def load_data():
    
    # CRIM     per capita crime rate by town
    # ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    # INDUS    proportion of non-retail business acres per town
    # CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    # NOX      nitric oxides concentration (parts per 10 million)
    # RM       average number of rooms per dwelling
    # AGE      proportion of owner-occupied units built prior to 1940
    # DIS      weighted distances to five Boston employment centres
    # RAD      index of accessibility to radial highways
    # TAX      full-value property-tax rate per $10,000
    # PTRATIO  pupil-teacher ratio by town
    # B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    # LSTAT    % lower status of the population
    # MEDV     Median value of owner-occupied homes in $1000's
    
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # now we split this into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # These are the Feature names based on the original dataset
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target # here MEDV is our target variable
    return df