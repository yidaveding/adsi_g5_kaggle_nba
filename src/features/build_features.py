import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def add_features_min_average(dfx, scaler=None, scale=True):
    
    # generate 7 new per minute average features by dividing stats by MIN. Only works for the NBA dataset
    # input: 
        # df: pandas data frame 
        # scale: boolean, if set to true, a scaler will be required
        # scaler: sklearn scaler is required if scale is set to True, only support Standard and MinMax scalers
    # return:
        # scaled data frame with new added features
    
    df = dfx.copy()
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


def add_off_def_features(offensive_features, defensive_features, df_scaledx):
    
    """add 2 x new features: offensive and defensive ratings:

    Parameters
    ----------
    offensive_features: Numpy Array
        Features related to offensive ratings
    defensive_features: Numpy Array
        Features related to defensive ratings 
    df_scaled: Pandas Dataframe
        Scaled data frame that includes all offensive and defensive features

    Returns
    -------
    df: Pandas Dataframe
        A new dataframe that adds the two new features
    """
    df_scaled = df_scaledx.copy()
    df_scaled['offensive_ratings'] = df_scaled[offensive_features].mean(axis=1)
    df_scaled['defensive_ratings'] = df_scaled[defensive_features].mean(axis=1)
    
    return df_scaled


def manual_splitter(dfx, offensive_avg, defensive_avg):
    """manually split data into 4 gropus according to high and low offensive and efensive ratings:

    Parameters
    ----------
    df: pandas dataframe
        data to be split
    offensive_avg: float
        offensive mean / median 
    defensive_avg: float
        defensive mean / median 

    Returns
    -------
    df: Pandas Dataframe
        A new dataframe with new column 'offensive_deffensive_cluster'
    """    
    
    df = dfx.copy()
        
    conditions = [
        (df['offensive_ratings']>=offensive_avg) & (df['defensive_ratings']>=defensive_avg),
        (df['offensive_ratings']>=offensive_avg) & (df['defensive_ratings']< defensive_avg),
        (df['offensive_ratings']< offensive_avg) & (df['defensive_ratings']>=defensive_avg),
        (df['offensive_ratings']< offensive_avg) & (df['defensive_ratings']< defensive_avg)
    ]
    
    values = ['ho_hd', 'ho_ld', 'lo_hd', 'lo_ld']

    df['offensive_deffensive_cluster'] = np.select(conditions, values)
    
    return df


def data_splitter(clr_model, data, label, off_feat, def_feat):
    """split data using cluster model, adding new features before spliting:

    Parameters
    ----------
    clr_model: clustering model
        clustering model
    data: pandas dataframe
        data to be split
    label: numpy array
        training label for data
    off_feat: Numpy array
        list of offensive features
    def_feat: Numpy array
        list of defensive features 

    Returns
    -------
    df: Pandas Dataframe
        A new dataframe with new column 'offensive_deffensive_cluster'
    """    
    
    # determine cluster group with KMmeans method
    obs = clr_model.predict(data)

    # combine data for later split
    data_split = data.copy()
    data_split['target'] = label
    data_split['cluster'] = obs

    # add offensive and defensive ratings
    data_split = add_off_def_features(off_feat, def_feat, data_split)

    # add per minute features
    data_split = add_features_min_average(data_split)

    # split data by cluster groups
    p1_clr = [2,3,4,6]
    p2_clr = [0,1,5,7]
    
    # split data
    data_p1 = data_split[data_split['cluster'].isin(p1_clr)]
    label_p1 = data_p1.pop('target')
    data_p2 = data_split[data_split['cluster'].isin(p2_clr)]
    label_p2 = data_p2.pop('target')
    
    return data_p1, label_p1, data_p2, label_p2