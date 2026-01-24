import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class PCADirectHARX:
    def __init__(self, volatility_measure:pd.DataFrame, exo_df:pd.DataFrame, weekly:int=5, monthly:int=22):
        self.vix_names = ['^VIX', '^VVIX']
        self.commodity_names = ['^MOVE', '^GVZ', '^OVX']
        self.data = volatility_measure.copy()
        self.days_by_period = {
            'weekly': weekly,
            'monthly': monthly
        }
        
        self.df = self._prepare_features()
        self.df = pd.concat([self.df, exo_df], axis=1).dropna()
        
        self.vix_idx = self._get_col_idx(self.df, self.vix_names)
        self.comm_idx = self._get_col_idx(self.df, self.commodity_names)
    
    def _prepare_features(self) -> pd.DataFrame:
        prev_day = self.data.shift(1)
        weekly = self.data.rolling(self.days_by_period['weekly']).mean().shift(1) #current volatility cannot rely on a rolling average using its own value 
        monthly = self.data.rolling(self.days_by_period['monthly']).mean().shift(1)
        col_names = ['volatility', 'prev_day', 'weekly_rolling_avg', 'monthly_rolling_avg']
        df = pd.concat([self.data, prev_day, weekly, monthly], axis=1)
        df.columns = col_names
        return df.dropna()
    
    def _get_col_idx(self, df:pd.DataFrame, col_names:list[str]) -> list[int]:
        idxs = [df.columns.get_loc(col) - 1 for col in col_names] #Deduct 1 due to a preprocessing step in backtest and fit.
        return idxs
    
    def _apply_pca(self, X_train:np.ndarray, X_test:np.ndarray, n_components:int) -> tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components=n_components)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_pca = pca.fit_transform(X_scaled) #(len(X_train), n_components)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled) #(len(X_test), n_components)
        
        return X_pca, X_test_pca
    
    def _pca_pipeline(self, X_train:np.ndarray, X_test:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        #Safe Modifications
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        #VIX PCA
        X_vix_train, X_vix_test = X_train[:, self.vix_idx], X_test[:, self.vix_idx]
        X_vix_train, X_vix_test = self._apply_pca(X_vix_train, X_vix_test, n_components=1)
        
        #Commodities PCA
        X_comm_train, X_comm_test = X_train[:, self.comm_idx], X_test[:, self.comm_idx]
        X_comm_train, X_comm_test = self._apply_pca(X_comm_train, X_comm_test, n_components=1)
        
        #Create X Matrices
        target_idx = self.vix_idx + self.comm_idx
        X_train = np.delete(X_train, target_idx, axis=1)
        X_train = np.column_stack((X_train, X_vix_train, X_comm_train)) #(len(X_train), initial_col-5+2)
        X_test = np.delete(X_test, target_idx, axis=1)
        X_test = np.column_stack((X_test, X_vix_test, X_comm_test))
        
        return X_train, X_test
    
    def _rolling_ols(self, X:np.ndarray, y:np.ndarray, window:int=1000) -> dict:
        lr = LinearRegression(fit_intercept=True)
        preds, betas, intercepts = [], [], []
        
        for k in range(window, len(y)):
            start_pt = k-window
            X_train = X[start_pt:k] #right side excludes k-th data point
            X_test = X[[k]]
            
            X_train, X_test = self._pca_pipeline(X_train, X_test)
            y_train = y[start_pt:k]
            model = lr.fit(X_train, y_train)
            
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
        col_names = list(X.columns[:3]) + ['market_vol_structure', 'cross_asset_vol_structure']
        betas = pd.DataFrame(results['betas'], index=X.index[window:], columns=col_names).add_prefix("beta_")
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
        X_idx = X.index
        col_names = list(X.columns[:3]) + ['market_vol_structure', 'cross_asset_vol_structure']
        X = X.values
        
        #VIX PCA
        vix_data = X[:, self.vix_idx]
        self.vix_pca = PCA(n_components=1)
        self.vix_scaler = StandardScaler()
        vix_scaled_data = self.vix_scaler.fit_transform(vix_data)
        vix_pca_data = self.vix_pca.fit_transform(vix_scaled_data)
        
        #Commodities PCA
        comm_data = X[:, self.comm_idx]
        self.comm_pca = PCA(n_components=1)
        self.comm_scaler = StandardScaler()
        comm_scaled_data = self.comm_scaler.fit_transform(comm_data)
        comm_pca_data = self.comm_pca.fit_transform(comm_scaled_data)
        
        #Construct X Matrix
        X = np.delete(X, self.vix_idx+self.comm_idx, axis=1)
        X = np.column_stack((X, vix_pca_data, comm_pca_data))
        
        y = self._build_target(horizon)
        y.columns = ['target']
        self.fitted_values = pd.DataFrame(X, index=X_idx, columns=col_names).join(y, how='inner').dropna()
        self.model = LinearRegression(fit_intercept=True).fit(X, self.fitted_values['target'].values)
    
    def forecast(self) -> np.ndarray[float]:
        """Uses the latest data and the fitted regressors to predict the next h-day average volatility."""
        X = self.df.iloc[[-1], 1:].values
        
        vix_pca_data = self.vix_pca.transform(
            self.vix_scaler.transform(X[:, self.vix_idx])
        )
        comm_pca_data = self.comm_pca.transform(
            self.comm_scaler.transform(X[:, self.comm_idx])
        )
        
        X = np.delete(X, self.vix_idx+self.comm_idx, axis=1)
        X = np.column_stack((X, vix_pca_data, comm_pca_data))
        
        pred = self.model.predict(X)[0]
        return pred
    
    @property
    def mse(self) -> float:
        """The average squared error between the HAR model and the target volatility measure, not the intraday Realized Volatility."""
        if not hasattr(self, "eval"):
            raise ValueError("Evaluate the model using .backtest() before calling this method.")
        
        squared_error = (self.eval.iloc[:, -1] - self.eval.iloc[:, -2])**2
        mse = squared_error.mean()
        return mse