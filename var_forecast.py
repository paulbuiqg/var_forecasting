# %%
# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.varmax import VARMAX

# %%
# Constants

VAR_LEVEL = 0.95
TRAINING = 0.8  # Proportion of data for initial inference

# %%
# Dataloading

data = pd.read_csv('/Users/paul/Code/IFT/asset-price-data-20240610.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data = data.sort_values('Date')
data = data.reset_index(drop=True)

data['BTC_return'] = np.log(data['BTC']) - np.log(data['BTC'].shift())
data['ETH_return'] = np.log(data['ETH']) - np.log(data['ETH'].shift())
data['SNT_return'] = np.log(data['SNT']) - np.log(data['SNT'].shift())
data['EUR_return'] = np.log(data['EUR']) - np.log(data['EUR'].shift())

data = data.dropna()

# Train-test split
data_train = data.iloc[:int(TRAINING * len(data))]
data_test = data.iloc[int(TRAINING * len(data)):]

data.head()

# %%
# Timeseries plot

plt.plot(data_train['Date'], data_train['BTC_return'], label='Training')
plt.plot(data_test['Date'], data_test['BTC_return'], label='Test')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Log-return')
plt.legend()
plt.title('BTC')
plt.show()

plt.plot(data_train['Date'], data_train['ETH_return'], label='Training')
plt.plot(data_test['Date'], data_test['ETH_return'], label='Test')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Log-return')
plt.legend()
plt.title('ETH')
plt.show()

plt.plot(data_train['Date'], data_train['SNT_return'], label='Training')
plt.plot(data_test['Date'], data_test['SNT_return'], label='Test')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Log-return')
plt.legend()
plt.title('SNT')
plt.show()

plt.plot(data_train['Date'], data_train['EUR_return'], label='Training')
plt.plot(data_test['Date'], data_test['EUR_return'], label='Test')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Log-return')
plt.legend()
plt.title('EUR')
plt.show()

# %%
# Historical VaR computation

# This dataset gathers VaR forecasts (one-day-ahead)
data_var = data_test.copy()[['Date']]

BTC_hist_var = []
ETH_hist_var = []
SNT_hist_var = []
EUR_hist_var = []

X_BTC_live = data_train['BTC_return'].values
X_ETH_live = data_train['ETH_return'].values
X_SNT_live = data_train['SNT_return'].values
X_EUR_live = data_train['EUR_return'].values

for i in range(len(data_test)):

    if i > 0:
        X_BTC_live = np.append(X_BTC_live, data_test['BTC_return'].values[i - 1])
        X_ETH_live = np.append(X_ETH_live, data_test['ETH_return'].values[i - 1])
        X_SNT_live = np.append(X_SNT_live, data_test['SNT_return'].values[i - 1])
        X_EUR_live = np.append(X_EUR_live, data_test['EUR_return'].values[i - 1])

    BTC_hist_var.append(np.quantile(X_BTC_live, 1 - VAR_LEVEL))
    ETH_hist_var.append(np.quantile(X_ETH_live, 1 - VAR_LEVEL))
    SNT_hist_var.append(np.quantile(X_SNT_live, 1 - VAR_LEVEL))
    EUR_hist_var.append(np.quantile(X_EUR_live, 1 - VAR_LEVEL))

data_var['BTC_hist_var'] = BTC_hist_var
data_var['ETH_hist_var'] = ETH_hist_var
data_var['SNT_hist_var'] = SNT_hist_var
data_var['EUR_hist_var'] = EUR_hist_var

# %%
# Model VaR computation

X_train = data_train[['BTC_return', 'ETH_return', 'SNT_return', 'EUR_return']].values
X_test = data_test[['BTC_return', 'ETH_return', 'SNT_return', 'EUR_return']].values
X_live = X_train

BTC_model_var = []
ETH_model_var = []
SNT_model_var = []
EUR_model_var = []

for i in range(len(data_test)):

    print(data_test['Date'].iloc[i])

    if i > 0:
        X_live = np.vstack((X_live, X_test[i,:]))

    model = VARMAX(X_live, order=(3, 3))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=1)
    forecast_ci = forecast.conf_int(alpha=(1 - VAR_LEVEL) * 2)

    BTC_model_var.append(forecast_ci[0, 0])
    ETH_model_var.append(forecast_ci[0, 1])
    SNT_model_var.append(forecast_ci[0, 2])
    EUR_model_var.append(forecast_ci[0, 3])

data_var['BTC_model_var'] = BTC_model_var
data_var['ETH_model_var'] = ETH_model_var
data_var['SNT_model_var'] = SNT_model_var
data_var['EUR_model_var'] = EUR_model_var

data_var.to_csv('VaR.csv', index=False)

# %%
# VaR violations

data_var['BTC_hist_var_violation'] = data_var['BTC_hist_var'] > data_test['BTC_return']
data_var['BTC_model_var_violation'] = data_var['BTC_model_var'] > data_test['BTC_return']
data_var['ETH_hist_var_violation'] = data_var['ETH_hist_var'] > data_test['ETH_return']
data_var['ETH_model_var_violation'] = data_var['ETH_model_var'] > data_test['ETH_return']
data_var['SNT_hist_var_violation'] = data_var['SNT_hist_var'] > data_test['SNT_return']
data_var['SNT_model_var_violation'] = data_var['SNT_model_var'] > data_test['SNT_return']
data_var['EUR_hist_var_violation'] = data_var['EUR_hist_var'] > data_test['EUR_return']
data_var['EUR_model_var_violation'] = data_var['EUR_model_var'] > data_test['EUR_return']

# %%
# VaR timeseries plot

def make_var_plot(data_test: pd.DataFrame, data_var: pd.DataFrame, asset: str):
    plt.plot(data_test['Date'], data_test[f'{asset}_return'], label='Observation')
    plt.plot(data_var['Date'], data_var[f'{asset}_hist_var'], label='Historical VaR',
             color='red', linestyle='dashed')
    plt.plot(data_var['Date'], data_var[f'{asset}_model_var'], label='Model VaR',
             color='purple', linestyle='dashed')
    plt.scatter(data_var.loc[data_var[f'{asset}_hist_var_violation'], 'Date'],
                data_test.loc[data_var[f'{asset}_hist_var_violation'], f'{asset}_return'],
                color='red', label='Historical VaR violation')
    plt.scatter(data_var.loc[data_var[f'{asset}_model_var_violation'], 'Date'],
                data_test.loc[data_var[f'{asset}_model_var_violation'], f'{asset}_return'],
                color='purple', label='Model VaR violation', marker="D", alpha=.5)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Log-return')
    plt.legend()
    plt.title(asset)
    plt.show()

make_var_plot(data_test, data_var, 'BTC')
make_var_plot(data_test, data_var, 'ETH')
make_var_plot(data_test, data_var, 'SNT')
make_var_plot(data_test, data_var, 'EUR')

# %%
