import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)

    

def load_split_store(filepath='data/', scaler=None, random_state=42, test_size=0.15, resample=False, replace_negatives=True):
    """Perform a number of data processing
              - load nba data, 
          - scale data, 
          - up sample, 
          - split data into train, val and test sets and 
          - store dataset as numpy array in data/processed folder 

    Parameters
    ----------
    filepath: Text String
        The data location (default: 'data/')
    scaler: sklearn scaler
        scaler to be used to scale data
    random_state: integer
        random state to be used in data splits
    test_size: double range between 0-1
        Propotion for test and validation sets
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    resample: Boolean
        if True, use SMOTE to resample data or not (default: True)
    replace_negatives: Boolean
        if True, replace all negative values with zero (default: True)

    Returns
    -------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    test_scaled: Numpy Array
        scaled test data
    """
    # load raw data        
    train_data = pd.read_csv(filepath + 'raw/train.csv')
    test_data = pd.read_csv(filepath + 'raw/test.csv')
    
    # replace negative values with zero
    if replace_negatives==True:
        train_data[train_data<0] = 0
        test_data[test_data<0] = 0
    
    # initial modifications:
    data_mod = train_data.copy()
        # drop ID columns
    data_mod = data_mod.drop(['Id_old', 'Id'], axis=1)
        # split target label
    data_target = data_mod.pop('TARGET_5Yrs')
        # load test data
    test_mod = test_data.copy().drop(['Id_old', 'Id'], axis=1)
    
    # scale data
    if scaler == None:
        data_scaled, test_scaled = data_mod, test_mod
    else:
        data_scaled = pd.DataFrame(scaler.fit_transform(data_mod), columns=data_mod.columns)
        test_scaled = pd.DataFrame(scaler.fit_transform(test_mod), columns=test_mod.columns)
    
    # check resample
    if resample == True:
        sm = SMOTE(random_state=random_state)
        data_resample, target_resample = sm.fit_resample(data_scaled, data_target)
        print('Balance of positive and negative classes (%):')
        print(target_resample.value_counts(normalize=True) * 100)
    else:
        data_resample, target_resample = data_scaled, data_target
        
    # trian test split
        # split test data out
    X_data, X_test, y_data, y_test = train_test_split(data_resample, target_resample, test_size=test_size, random_state=random_state)
        # split validation data out
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state)
    
    # save data into location: data/processed
    save_sets(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, path='data/processed/')
    
    return X_train, y_train, X_val, y_val, X_test, y_test, test_scaled
    

    
    
