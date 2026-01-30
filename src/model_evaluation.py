import pandas as pd
import numpy as np
from arch.bootstrap import MCS

def mean_squared_error(actual:pd.Series, forecast:pd.Series) -> float:
    squared_error = (actual - forecast)**2
    return np.mean(squared_error)

def qlike(actual:pd.Series, forecast:pd.Series, is_averaged:bool=True) -> pd.Series|float:
    #Omit 0 values as QLIKE would be undefined
    mask = (actual > 0) & (forecast > 0)
    actual = actual[mask]
    forecast = forecast[mask]
    
    diff = actual/forecast
    error = (diff - np.log(diff) - 1)
    if is_averaged:
        error = np.mean(error)
    return error

def mcs_test(loss_array:np.ndarray, alpha:float=0.05, sample_length:int=10000, seed:int=None) -> list[int]:
    mcs = MCS(loss_array, size=alpha, reps=sample_length, seed=seed)
    mcs.compute()
    return mcs.included