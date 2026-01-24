import numpy as np
import pandas as pd
from arch import arch_model
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

class RollingGARCH:
    def __init__(self, log_returns_df:pd.DataFrame):
        self.df = log_returns_df.copy()
    
    def _rolling_garch(self, y:np.ndarray, window:int=1000) -> dict:
        preds, coeffs = [], []
        for k in range(window, len(y)):
            y_train = y[k-window:k]
            model = arch_model(y_train, mean='constant', vol='GARCH', p=1, q=1, dist='t').fit(disp=False)
            pred = model.forecast(horizon=1)
            pred = np.sqrt(pred.variance.values[0][0]) #Extract the number
            
            preds.append(pred)
            coeffs.append(model.params.to_dict())
        
        results = {
            'preds': preds,
            'coeffs': coeffs
        }
        return results
    
    def backtest(self, window:int=1000) -> pd.DataFrame:
        results = self._rolling_garch(self.df.values, window=window)
        coeffs = results['coeffs']
        preds = results['preds']
        
        self.eval = pd.DataFrame(coeffs)
        self.eval['preds'] = np.asarray(preds, dtype=float)
        self.eval.index = self.df.index[window:]
        return self.eval
    
    def fit(self, window:int=1000):
        self.window = window
        self.fitted_values = self.df.iloc[-window:]
        y = self.df.iloc[-window:]
        self.model = arch_model(y, mean='constant', vol='GARCH', p=1, q=1, dist='t').fit(disp=False)
    
    def forecast(self, steps:int=1) -> np.ndarray:
        if steps < 1:
            raise ValueError("Number of steps must be more than or equal to 1.")
        pred = self.model.forecast(horizon=steps)
        pred = pred.variance.iloc[-1].values.flatten()
        return np.sqrt(pred)

class OptimizedRGARCH(RollingGARCH):
    def __init__(self, log_returns_df:pd.DataFrame):
        super().__init__(log_returns_df)
    
    @staticmethod
    def _fit_predict(y_window) -> tuple[np.float64, dict]:
        model = arch_model(y_window, mean='constant', vol='GARCH', p=1, q=1, dist='t').fit(disp=False)
        pred = model.forecast(horizon=1)
        pred = np.sqrt(pred.variance.values[0][0]) #Extract the number
        coeff = model.params.to_dict()
        return (pred, coeff)
    
    def _rolling_garch(self, y:np.ndarray, window:int=1000) -> dict:
        results = Parallel(n_jobs=-2)(
            delayed(OptimizedRGARCH._fit_predict)(y[k-window:k])
            for k in range(window, len(y))
        )
        
        preds, coeffs = zip(*results)
        results = {
            'preds': preds,
            'coeffs': coeffs
        }
        return results

class RecursiveHAR:
    """
    RecursiveHAR implements a low-frequency Heterogeneous Autoregressive (HAR) model 
    to forecast realized volatility using daily, weekly, and monthly lagged components.
    For the forecasting method, it uses the rolling window to update its parameters.
    Subsequently, it uses the forecasted value to re-fit and forecast the next timestep.

    Input:
        volatility_measure : A time series of realized volatility estimates (e.g. Rogers-Satchell, Garman-Klass, Parkinson).
        weekly : Number of observations to use for the weekly rolling average (typically 5 for trading days).
        monthly : Number of observations to use for the monthly rolling average (typically 22 for trading days).
    """
    def __init__(self, volatility_measure:pd.DataFrame, weekly:int=5, monthly:int=22):
        self.data = volatility_measure.copy()
        self.days_by_period = {
            'weekly': weekly,
            'monthly': monthly
        }
        
        self.df = self._prepare_features()
    
    def _prepare_features(self) -> pd.DataFrame:
        prev_day = self.data.shift(1)
        weekly = self.data.rolling(self.days_by_period['weekly']).mean().shift(1) #current volatility cannot rely on a rolling average using its own value 
        monthly = self.data.rolling(self.days_by_period['monthly']).mean().shift(1)
        col_names = ['volatility', 'prev_day', 'weekly_rolling_avg', 'monthly_rolling_avg']
        df = pd.concat([self.data, prev_day, weekly, monthly], axis=1)
        df.columns = col_names
        return df.dropna()
    
    def _rolling_ols(self, X:np.ndarray, y:np.ndarray, window:int=1000) -> dict:
        lr = LinearRegression(fit_intercept=True)
        preds, betas, intercepts = [], [], []
        
        for k in range(window, len(y)):
            X_train = X[k-window:k] #right side excludes k-th data point
            y_train = y[k-window:k]
            model = lr.fit(X_train, y_train)
            
            X_test = X[[k]]
            y_pred = model.predict(X_test)
            preds.append(y_pred[0])
            betas.append(model.coef_)
            intercepts.append(model.intercept_)
        
        results = {
            'preds': preds,
            'betas': betas,
            'intercepts': intercepts
        }
        
        return results
    
    def backtest(self, window:int=1000) -> pd.DataFrame:
        X = self.df.iloc[:, 1:].values
        y = self.df.iloc[:, 0].values
        results = self._rolling_ols(X, y, window)
        
        #Create Dataframe
        preds = pd.Series(results['preds'], index=self.df.index[window:], name='y_hat')
        betas = pd.DataFrame(results['betas'], index=self.df.index[window:], columns=self.df.columns[1:])
        betas = betas.add_prefix("beta_")
        intercepts = pd.Series(results['intercepts'], index=self.df.index[window:], name='intercept')
        self.eval = pd.concat([betas, intercepts, preds, self.df.iloc[window:, 0]], axis=1)
        return self.eval
    
    def fit(self, window:int=1000):
        self.window = window
        self.fitted_values = self.df.iloc[-window:, 1:]
        y = self.df.iloc[-window:, 0]
        self.model = LinearRegression(fit_intercept=True).fit(self.fitted_values.values, y.values)
    
    def _process_next_day_features(self, latest_data:np.ndarray) -> np.ndarray:
        prev_day = latest_data[-1]
        weekly = latest_data[-self.days_by_period['weekly']:].mean()
        monthly = latest_data[-self.days_by_period['monthly']:].mean()
        return np.array([[prev_day, weekly, monthly]]) #(1, 3)
    
    def forecast(self, steps:int=1) -> np.ndarray:
        if steps < 1:
            raise ValueError("Number of steps must be more than or equal to 1.")
        
        latest_data = self.data.iloc[-self.days_by_period['monthly']:].values.flatten()
        preds = []
        for _ in range(steps):
            X = self._process_next_day_features(latest_data)
            pred = self.model.predict(X)[0]
            preds.append(pred)
            
            #Update latest data
            latest_data[:-1] = latest_data[1:]
            latest_data[-1] = pred
        
        return np.array(preds)
    
    @property
    def mse(self) -> float:
        """The average squared error between the HAR model and the target volatility measure, not the intraday Realized Volatility."""
        if not hasattr(self, "eval"):
            raise ValueError("Evaluate the model using .backtest() before calling this method.")
        
        squared_error = (self.eval.iloc[:, -1] - self.eval.iloc[:, -2])**2
        mse = squared_error.mean()
        return mse

class OptimizedRecursiveHAR(RecursiveHAR):
    def __init__(self, volatility_measure:pd.DataFrame, weekly:int=5, monthly:int=22):
        super().__init__(volatility_measure, weekly, monthly)
    
    @staticmethod
    def _fit_predict(X_window, y_window, X_test_row) -> tuple[float, np.ndarray, float]:
        model = LinearRegression().fit(X_window, y_window)
        return model.predict(X_test_row)[0], model.coef_, model.intercept_
    
    def _rolling_ols(self, X:np.ndarray, y:np.ndarray, window:int) -> dict:
        results = Parallel(n_jobs=-2)(
            delayed(OptimizedRecursiveHAR._fit_predict)(
                X[k - window:k],
                y[k - window:k],
                X[[k]]
            )
            for k in range(window, len(y))
        )
        preds, betas, intercepts = zip(*results)
        
        results = {
            'preds': preds,
            'betas': betas,
            'intercepts': intercepts
        }
        
        return results

class DirectHAR(RecursiveHAR):
    """
    DirectHAR implements a low-frequency Heterogeneous Autoregressive (HAR) model 
    to forecast realized volatility using daily, weekly, and monthly lagged components.
    For the forecasting method, it uses the rolling window to update its parameters.

    Key differences vs RecursiveHAR:
    - Uses DIRECT multi-horizon forecasting
    - Never feeds forecasted values back into regressors
    - Redefines the dependent variable as a future rolling average
    """
    def __init__(self, volatility_measure:pd.DataFrame, weekly:int=5, monthly:int=22):
        super().__init__(volatility_measure, weekly, monthly)
    
    def _build_target(self, horizon:int) -> pd.Series:
        """Redefine the dependent variable as the future h-average volatility"""
        if horizon < 1:
            raise ValueError("Horizon must be more than or equal to 1.")
        
        new_target = self.data.rolling(horizon).mean()
        new_target = new_target.shift(-horizon+1) #Use the data today, to predict the average h-day future volatility
        return new_target.dropna()
    
    def backtest(self, horizon:int, window:int=1000) -> pd.DataFrame:
        target = self._build_target(horizon=horizon)
        df = self.df.iloc[:, 1:].join(target, how="inner") #Align Index
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y.name = 'target'
        results = self._rolling_ols(X.values, y.values, window)
        
        #Create Dataframe
        betas = pd.DataFrame(results['betas'], index=X.index[window:], columns=X.columns).add_prefix("beta_")
        intercepts = pd.Series(results['intercepts'], index=X.index[window:], name='intercept')
        preds = pd.Series(results['preds'], index=X.index[window:], name='y_hat')
        self.eval = pd.concat([betas, intercepts, preds, y.iloc[window:]], axis=1)
        return self.eval
    
    def fit(self, horizon:int, window:int=1000):
        """
        Fit the HAR model on the latest window. If we are forecasting for more than one timestep, the target will be redefined to be the future h-average volatility.
            
        Alignment logic:
            - Features X are built from rows up to time t
            - Targets y correspond to outcomes at time t + h
            - To avoid look-ahead bias, the effective training window must be shifted backward by (horizon - 1) observations.
        
        Input:
            horizon: Forecast horizon, where 1 is the standard one-step, and 5 is the 5-day future-averaged target.
            window: Number of aligned observations used for the rolling estimation window.
        """
        start_pt = -window-(horizon-1)
        X = self.df.iloc[start_pt:-horizon+1, 1:] if horizon > 1 else self.df.iloc[start_pt:, 1:]
        y = self._build_target(horizon)
        y.columns = ['target']
        self.fitted_values = X.join(y, how='inner').dropna()
        self.model = LinearRegression(fit_intercept=True).fit(X.values, self.fitted_values['target'].values)
    
    def forecast(self) -> np.ndarray[float]:
        """Uses the latest data and the fitted regressors to predict the next h-day average volatility."""
        X = self.df.iloc[[-1], 1:].values
        pred = self.model.predict(X)[0]
        return pred

class DirectHARX(DirectHAR):
    """
    Ensure that the index of volatility_measure and exo_df are both of datetime type.
    """
    def __init__(self, volatility_measure:pd.DataFrame, exo_df:pd.DataFrame, weekly:int=5, monthly:int=22):
        super().__init__(volatility_measure, weekly, monthly)
        self.df = pd.concat([self.df, exo_df], axis=1).dropna()