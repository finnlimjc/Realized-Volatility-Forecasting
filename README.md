# Instructions
1. Create a virtual environment:
```sh
# Open a terminal and navigate to your project folder
cd myproject

# Create the .venv folder
python -m venv .venv
```

2. Activate the virtual environment:
```sh
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and LinuxS
source .venv/bin/activate
```

3. Install packages in the environment:
```sh
python -m pip install -r requirements.txt
```

4. Create a secrets.env file in the root folder with the key and secret retrieved from https://alpaca.markets/
```sh
ALPACA_KEY=API_KEY
ALPACA_SECRET=SECRET_KEY
```

6. Run the dashboard:
```sh
streamlit run app.py
```

7. Alternatively, if you have already completed the virtual environment and packages installation, run the appropriate commands as follows:
```sh
cd myproject
.venv\Scripts\activate
python -m streamlit run app.py
```

# Dashboard
Note that the goal of this project was to learn about volatility forecasting models, model evaluation methods, and the exploration of including exogenous variables. In the dashboard, a few assumptions were made to simplify the exploration of the volatility models.
- GARCH(1, 1) is the best parameter for GARCH.
- '^VIX', '^VVIX', '^MOVE', '^GVZ', '^OVX' sufficiently explains macro-factors that affect market volatility.
- MCS at the 5% significance level with a bootstrap length of 10,000 and the default block size of $\sqrt{T}$ is valid.

<img width="2000" height="879" alt="image" src="https://github.com/user-attachments/assets/7d7db462-fe07-4ad1-8882-22763f2f6859" />

# About the Project
This project focuses on short-horizon volatility forecasting using both realized volatility and return-based models. Realized volatility is proxied using the Garman–Klass estimator, which incorporates intraday price information to provide a more efficient measure than squared returns. The realized volatility series is modeled using the Heterogeneous Autoregressive (HAR) framework to capture multi-scale persistence in volatility dynamics. In parallel, a GARCH model is fitted to log returns to provide a traditional return-based volatility benchmark. All models are estimated using a rolling window of 1,000 observations and evaluated under a one-step-ahead forecasting setup. The forecasting horizon can be naturally extended to two- or three-step-ahead forecasts through a direct forecasting approach, allowing flexibility in evaluating short-term volatility dynamics across different horizons. However, the focus for this project was the evaluation of models using a one-step forecast.

In addition to baseline HAR and GARCH specifications, we extend the realized volatility framework by incorporating exogenous variables into the HAR model (HAR-X) to capture information from macro-factors that affect the market volatility. Principal component analysis (PCA) is conducted separately on the VIX–VVIX and MOVE–GVZ–OVX variable groups, with the leading principal component from each group used as the exogenous factor to capture the dominant common variation in volatility dynamics. Model performance is evaluated using the QLIKE loss function and mean squared error (MSE), which are standard metrics in the volatility forecasting literature. To formally assess relative model performance, we further employ the Model Confidence Set (MCS) procedure to identify the subset of statistically superior models.

Below are details of the exogenous variables used:
- **^VIX**: Implied volatility of S&P 500 options; a forward-looking measure of market uncertainty and the primary benchmark for equity market volatility expectations.
- **^VVIX**: Implied volatility of VIX options; captures uncertainty about future volatility itself and is often interpreted as a proxy for volatility risk and regime uncertainty.
- **^MOVE**: Implied volatility of U.S. Treasury yields; reflects interest rate uncertainty and macro-financial stress, which often spills over into equity volatility.
- **^GVZ**: Implied volatility of gold prices; serves as a proxy for safe-haven demand, inflation concerns, and risk-off sentiment.
- **^OVX**: Implied volatility of crude oil prices; captures energy market uncertainty and geopolitical or supply-driven risk that can transmit to broader financial markets.

## Volatility Proxies
**Parkinson Volatility**

$$P_t = \sqrt{\frac{1}{4log(2)}\left[log\left(\frac{h_t}{l_t}\right)\right]^2}$$

**Garman-Klass Volatility**

$$GK_t = \sqrt{\frac{1}{2}\left[log\left(\frac{h_t}{l_t}\right)\right]^2 - (2log(2) - 1)\left[log\left(\frac{c_t}{o_t}\right)\right]^2}$$

**Rogers-Satchell Volatility**

$$RS_t = \sqrt{log\left(\frac{h_t}{o_t}\right)log\left(\frac{h_t}{c_t}\right) + log\left(\frac{l_t}{o_t}\right)log\left(\frac{l_t}{c_t}\right)}$$

**Average of the Three**

$$RB_t = \frac{P_t + GK_t + RS_t}{3}$$

## GARCH
$$y_t = \mu + \epsilon_t$$
$$\sigma^2_t = \omega + \alpha\epsilon^2_{t-1} + \beta\sigma^2_{t-1}$$
$$\mu = \frac{\omega}{1-\alpha-\beta}$$

## HAR
$$RV_t = \beta_0 +  \beta_1RV_{t-1} + \beta_2\frac{1}{W}\sum^W_{i=1}RV_{t-i} + \beta_3\frac{1}{M}\sum^M_{i=1}RV_{t-i} + \epsilon_t $$

where $W=5$ and $M=22$. However, this could change for cryptocurrency as the market also trades on weekends, hence we get $W=7$ and $M=30$.

## Model Evaluation
$$MSE=\frac{1}{T}\sum^T_{t=1}\left(RV_{t+h-1|t} - F_t\right)^2$$
$$QLIKE = \frac{1}{T}\sum^T_{t=1}\left(\frac{RV_{t+h-1|t}}{F_t} - log\frac{RV_{t+h-1|t}}{F_t}-1\right)$$

## Multi-Step Forecast
Note that a multi-step forecast version was implemented for educational purposes. However, the goal of this project was a one-step ahead forecast, and hence, this was not explored:

$$RV_{t+h-1|t} = \frac{1}{h}\sum^{t+h-1}_{i=t}RV_i, \quad h \geq 1$$

# References
- Clements, Adam and Preve, Daniel P. A. and Tee, Clarence, Harvesting the HAR-X Volatility Model (February 20, 2024). Available at SSRN: https://ssrn.com/abstract=4733597 or http://dx.doi.org/10.2139/ssrn.4733597
- Hansen, Peter Reinhard; Lunde, Asger; Nason, James M. (2003) : Choosing the best volatility models: The model confidence set approach, Working Paper, No. 2003-05, Brown University, Department of Economics, Providence, RI
