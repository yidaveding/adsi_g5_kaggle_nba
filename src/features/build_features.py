import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def add_features_min_average(df, scaler=None, scale=True):
    
    # generate 7 new per minute average features by dividing stats by MIN. Only works for the NBA dataset
    # input: 
        # df: pandas data frame 
        # scale: boolean, if set to true, a scaler will be required
        # scaler: sklearn scaler is required if scale is set to True, only support Standard and MinMax scalers
    # return:
        # scaled data frame with new added features
       
    df['PTS_pm'] = df.PTS / df.MIN
    df['FTM_pm'] = df.FTM / df.MIN
    df['REB_pm'] = df.REB / df.MIN
    df['AST_pm'] = df.AST / df.MIN
    df['STL_pm'] = df.STL / df.MIN
    df['BLK_pm'] = df.BLK / df.MIN
    df['TOV_pm'] = df.TOV / df.MIN
    
    if scale:
        if scaler==None:
            df = df
        else:
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df