import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import date
from dotenv import load_dotenv

from src.data_io import *
from src.model_evaluation import *
from src.models import *
from src.pca_harx import *
from src.volatility_estimates import *

class ParamsSelector:
    def select_stock_info(self) -> str:
        st.subheader("💰 Ticker Information")
        symbol = st.text_input("Yahoo Finance Ticker Symbol", value='SPY')
        return str(symbol)
    
    def select_date(self) -> tuple[str, str]:
        st.subheader("📅 Select Date Range")
        default_start = date(2010, 1, 1)
        default_end = date(2025, 12, 31)
        start_date = st.date_input("Start Date", default_start, min_value=default_start, max_value=default_end)
        end_date = st.date_input("End Date", default_end, min_value=default_start, max_value=default_end)
        
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return None
        
        # For YahooFinance
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        
        return (start_date, end_date)
    
    def select_volatility_measure(self) -> str:
        st.subheader("📋 Realized Volatility Proxy")
        
        valid_names = ('garman_klass', 'parkinson', 'rogers_satchell', 'average')
        volatility_measure = st.selectbox("Volatility Measure", valid_names, index=0)
        return volatility_measure
    
    def render(self) -> tuple[dict, str]:
        with st.sidebar:
            st.header("⚙️ Parameter Selector")
            symbol = self.select_stock_info()
            start_date, end_date = self.select_date()
            
            st.divider()
            volatility_measure = self.select_volatility_measure()

            yf_params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }
            
            return (yf_params, volatility_measure)

# --------------------------------------------------- DATA ------------------------------------------------------
def calculate_volatility_measure(df:pd.DataFrame, volatility_measure:str='garman_klass') -> pd.Series:
    df = df.copy()
    vm = VolatilityMeasures(df)
    
    valid_names = ('garman_klass', 'parkinson', 'rogers_satchell', 'average')
    if volatility_measure not in valid_names:
        raise ValueError(f'Volatility Measure must be one of the valid names: {valid_names}')
    
    if volatility_measure == valid_names[0]:
        result = vm.garman_klass_volatility(df['Close'], df['High'], df['Low'], df['Open'])
    elif volatility_measure == valid_names[1]:
        result = vm.parkinson_volatility(df['High'], df['Low'])
    elif volatility_measure == valid_names[2]:
        result = vm.rogers_satchell_volatility(df['Close'], df['High'], df['Low'], df['Open'])
    elif volatility_measure == valid_names[3]:
        result = vm.average_volatility(df['Close'], df['High'], df['Low'], df['Open'])
    
    return result

@st.cache_data
def get_training_data(symbol:str, start_date:str, end_date:str, volatility_measure:str='garman_klass') -> pd.DataFrame:
    params = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'interval': '1d'
    }
    yf = YahooFinance(**params)
    df = yf.pipeline()
    df[volatility_measure] = calculate_volatility_measure(df, volatility_measure)
    df.set_index('date', drop=True, inplace=True)
    df = df.sort_index(ascending=True)
    return df

@st.cache_data
def get_exo_data(start_date:str, end_date:str, date_format:str='%d/%m/%Y') -> pd.DataFrame:
    exo_symbols = ['^VIX', '^VVIX', '^MOVE', '^GVZ', '^OVX']
    exo_df = []
    
    for s in exo_symbols:
        params = {
            'symbol': s,
            'start_date': start_date,
            'end_date': end_date,
            'interval': '1d'
        }
        tmp = YahooFinance(**params)
        tmp_df = tmp.pipeline()
        
        tmp_df = tmp_df.loc[:, ['date', 'Close']]
        tmp_df.set_index('date', drop=True, inplace=True)
        tmp_df.index = pd.to_datetime(tmp_df.index, format=date_format)
        tmp_df = tmp_df.rename(columns={'Close': s})
        
        exo_df.append(tmp_df)
    
    exo_df = pd.concat(exo_df, axis=1).sort_index(ascending=True).ffill().dropna()
    scaled_exo_df = np.log(exo_df/ exo_df.shift(1)).replace([np.inf, -np.inf], 0).dropna()
    return scaled_exo_df

@st.cache_data
def get_target_data(symbol:str, start_date:str, end_date:str, api_key:str, secret_key:str) -> pd.DataFrame:
    alpaca_client = AlpacaStockData(api_key=api_key, secret_key=secret_key)
    intraday_df = alpaca_client.get_stock_bar(symbol=symbol, start=start_date, end=end_date)
    target_rv = alpaca_client.intraday_rv(intraday_df)
    return target_rv

# --------------------------------------------------- MODEL------------------------------------------------------
def backtest_garch(df:pd.DataFrame, log_return_col:str='log_return', is_pct:bool=False) -> pd.DataFrame:
    log_returns = df.loc[:, log_return_col].dropna()
    if not is_pct:
        log_returns *= 100
    
    r_garch = OptimizedRGARCH(log_returns)
    backtest = r_garch.backtest()
    
    if not is_pct:
        backtest['preds'] /= 100
    return backtest

def backtest_har(df:pd.DataFrame, volatility_measure:str='garman_klass') -> pd.DataFrame:
    vol = df.loc[:, [volatility_measure]]
    har = DirectHAR(vol)
    backtest = har.backtest(horizon=1, window=1000)
    return backtest

def backtest_harx(df:pd.DataFrame, scaled_exo_df:pd.DataFrame, volatility_measure:str='garman_klass') -> pd.DataFrame:
    vol = df.loc[:, [volatility_measure]]
    harx = PCADirectHARX(vol, scaled_exo_df)
    backtest = harx.backtest(horizon=1, window=1000)
    return backtest

@st.cache_data
def model_fit(df:pd.DataFrame, exo_df:pd.DataFrame, volatility_measure:str='garman_klass') -> pd.DataFrame:
    garch = backtest_garch(df)['preds']
    har = backtest_har(df, volatility_measure)['y_hat']
    harx = backtest_harx(df, exo_df, volatility_measure)['y_hat']
    
    results = {
        'RGARCH-t': garch,
        'HAR': har,
        'PCA-HAR-X': harx
    }
    
    backtest_df = pd.concat(results, axis=1, join='inner')
    backtest_df['RGARCH-HAR'] = backtest_df[['RGARCH-t', 'HAR']].mean(axis=1)
    backtest_df['RGARCH-HAR-X'] = backtest_df[['RGARCH-t', 'PCA-HAR-X']].mean(axis=1)
    return backtest_df

# --------------------------------------------------- EVALUATION---------------------------------------------------
@st.cache_data
def model_qlike(plot_df:pd.DataFrame) -> pd.DataFrame:
    forecast_df = plot_df.iloc[:, :-1]
    actual = plot_df.iloc[:, -1]
    qlike_df = forecast_df.apply(lambda col: qlike(actual, col, is_averaged=False), axis=0).dropna() #Compare only when all cols have a value
    return qlike_df

def get_mse(plot_df:pd.DataFrame) -> pd.Series:
    forecast_df = plot_df.iloc[:, :-1]
    actual = plot_df.iloc[:, -1]
    mse = forecast_df.apply(lambda col: mean_squared_error(actual, col), axis=0)
    return mse

def get_eval_df(plot_df:pd.DataFrame) -> pd.DataFrame:
    qlike_df = model_qlike(plot_df)
    mcs_included = mcs_test(qlike_df.values, alpha=0.05, sample_length=10000, seed=123)
    qlike_mean = qlike_df.mean()
    mse = get_mse(plot_df)
    
    included_flags = pd.Series(False, index=qlike_df.columns)
    included_flags.iloc[mcs_included] = True

    eval_df = pd.DataFrame({
        'QLIKE': qlike_mean,
        'MSE': mse,
        'MCS': included_flags
    }, index=plot_df.columns[:-1])
    return eval_df

# --------------------------------------------------- PLOTS---------------------------------------------------
def plot_line_chart(plot_df:pd.DataFrame, target_model:str, actual_col:str='realized_volatility') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8,2))
    sns.lineplot(plot_df[actual_col], label='Actual', alpha=0.7, ax=ax)
    sns.lineplot(plot_df[target_model], label='Forecast', linestyle='--', alpha=0.7, ax=ax)
    ax.legend()
    ax.set_ylabel('Realized Volatility')
    ax.set_title("Forecast vs Actual Volatility")
    return fig

def plot_distribution(plot_df:pd.DataFrame, target_model:str, actual_col:str='realized_volatility') -> plt.Figure:
    fig, ax = plt.subplots()
    sns.histplot(plot_df[actual_col].values, bins=50, stat='density', alpha=0.4, label='Actual', ax=ax)
    sns.histplot(plot_df[target_model].values, bins=50, stat='density', alpha=0.4, label='Forecast', ax=ax)
    ax.legend()
    ax.set_title("Distribution of Realized Volatility")
    return fig

def plot_scatter(plot_df:pd.DataFrame, target_model:str, actual_col:str='realized_volatility') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10,5))
    actual = plot_df[actual_col]
    forecast = plot_df[target_model]
    sns.scatterplot(x=actual, y=forecast, ax=ax, alpha=0.4)
    
    #45-degree line
    min_val = min(actual.min(), forecast.min())
    max_val = max(actual.max(), forecast.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Actual Realized Volatility')
    ax.set_ylabel('Forecasted Volatility')
    ax.set_title('Deviation from Optimal Line')
    return fig

if __name__ == '__main__':
    # Load Secrets
    load_dotenv("secrets.env")
    ALPACA_KEY = os.getenv("ALPACA_KEY")
    ALPACA_SECRET = os.getenv("ALPACA_SECRET")
    
    # Initial Set Up
    st.set_page_config(layout="wide")
    st.header("📊 Realized Volatility Forecasting Model")
    
    #Parameter Selection
    selector = ParamsSelector()
    yf_params, volatility_measure = selector.render()
    
    #Data Collection
    df = get_training_data(**yf_params, volatility_measure=volatility_measure)
    target_df = get_target_data(**yf_params, api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    scaled_exo_df = get_exo_data(start_date=yf_params['start_date'], end_date=yf_params['end_date'])
    
    #Model Training & Evaluation
    backtest_df = model_fit(df, scaled_exo_df, volatility_measure)
    plot_df = pd.concat([backtest_df, target_df], axis=1, join='inner') #Align indexes
    eval_df = get_eval_df(plot_df)
    
    #Model Selection
    target_model = st.selectbox("Select model to plot: ", options=plot_df.columns[:-1].tolist())
    
    #Plot
    col1a, col1b = st.columns((0.6, 0.4))
    with col1a:
        line_comparison = plot_line_chart(plot_df, target_model)
        st.pyplot(line_comparison)
        scatter = plot_scatter(plot_df, target_model)
        st.pyplot(scatter)
        
    with col1b:
        hist = plot_distribution(plot_df, target_model)
        st.pyplot(hist)
        st.dataframe(eval_df, width='stretch')