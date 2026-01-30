import pandas as pd
import numpy as np
from arch.bootstrap import MCS

def mean_squared_error(actual:pd.Series, forecast:pd.Series) -> float:
    squared_error = (actual - forecast)**2
    return np.mean(squared_error)

def qlike(actual:pd.Series, forecast:pd.Series, is_averaged:bool=True) -> float:
    #For stability
    mask = (actual > 0) & (forecast > 0)
    actual = actual[mask]
    forecast = forecast[mask]
    
    diff = actual/forecast
    error = (diff - np.log(diff) - 1)
    if is_averaged:
        error = np.mean(error)
    return error

def create_loss_df(actual:list[pd.Series], forecast:list[pd.Series], col_names:list[str], loss_fn:function) -> pd.DataFrame:
    if not (len(actual) == len(forecast) == len(col_names)):
        raise ValueError("Actual, Forecast, and col_names must have the same length.")
    
    loss_series = []
    for a, f in zip(actual, forecast):
        loss = loss_fn(actual=a, forecast=f, is_averaged=False)
        loss = pd.Series(loss, index=a.index)
        loss_series.append(loss)
    
    loss_df = pd.concat(loss_series, axis=1, join='inner').dropna() #Align the indexes to compare models fairly
    loss_df.columns = col_names
    return loss_df

def mcs_test(loss_array:np.ndarray, alpha:float=0.05, sample_length:int=10000, seed:int=None) -> list[int]:
    mcs = MCS(loss_array, size=alpha, reps=sample_length, seed=seed)
    mcs.compute()
    return mcs.included