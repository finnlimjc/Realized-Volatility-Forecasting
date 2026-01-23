import pandas as pd
import numpy as np

def format_yf_df(price_df:pd.DataFrame, col_name:str='Adj Close', date_format:str='%d/%m/%Y') -> pd.DataFrame:
    df = price_df.loc[:, ['date', col_name]]
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index, format=date_format)
    return df

def momentum_features(df:pd.DataFrame, window:int=22) -> pd.DataFrame:
    monthly_log_price = np.log(df[['Adj Close']])
    monthly_log_price.columns = [f'{window}_momentum']
    return monthly_log_price.diff(window)