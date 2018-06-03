import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import mixture as mix
from sklearn.preprocessing import StandardScaler

import Indicators as ind

ticker = 'TSLA'
ticker = 'stocks/A/AB'
df = pd.read_csv('db' + '/{0}.csv'.format(ticker), parse_dates = True)
df.set_index('Date', inplace = True)

if 'Symbol' in df.columns:
    df = df.drop('Symbol', axis = 1)

df = df.drop('Volume', axis = 1)

n = 10
t = 0.8
split = int(t*len(df))

df['high'] = df['High'].shift(1)
df['low'] = df['Low'].shift(1)
df['close'] = df['Close'].shift(1)

ind.RSI(df,n)
ind.MA(df,n)
ind.ADX(df, 10, 5)

df.rename(columns = {'RSI_10':'RSI', 'MA_10':'SMA', 'ADX_10_5':'ADX'}, inplace = True)

df['Corr'] = df['SMA'].rolling(window = n).corr(df['Close']) # Correlation
# df['SAR']

df['Return'] = np.log(df['Open']/df['Open'].shift(1))

df.dropna(inplace = True)


# * ---------- Standard Scaler ----------

ss = StandardScaler()
unsup = mix.GaussianMixture(n_components = 4, covariance_type = 'spherical', n_init = 100, random_state = 42)

df = df.drop(['High', 'Low', 'Close'], axis = 1)

unsup.fit(np.reshape(ss.fit_transform(df[:split]), (-1, df.shape[1])))

regime = unsup.predict(np.reshape(ss.fit_transform(df[split:]), (-1, df.shape[1])))

Regimes = pd.DataFrame(regime, columns = ['Regime'], index = df[split:].index)\
                        .join(df[split:], how = 'inner')\
                        .assign(market_cu_return = df[split:]\
                            .Return.cumsum())\
                            .reset_index(drop = False)\
                            .rename(columns = {'index':'Date'})

# order = [0,1,2,3]
# fig = sns.FacetGrid(data = Regimes, hue = 'Regime', hue_order = order, aspect = 2, size = 4)
# fig.map(plt.scatter, 'Date', 'market_cu_return', s = 4).add_legend()
# plt.show()



ss1 = StandardScaler()
columns = Regimes.columns.drop(['Regime', 'Date'])
Regimes[columns] = ss1.fit_transform(Regimes[columns])
Regimes['signal'] = 0
Regimes.loc[Regimes['Return'] > 0, 'Signal'] = 1
Regimes.loc[Regimes['Return'] < 0, 'Signal'] = -1

clf = SVC()

split2 = int(.8*len(Regimes))

X = Regimes.drop(['Signal','Return','market_cu_return','Date'], axis = 1)
y = Regimes['Signal']

clf.fit(X[:split2], y[:split2])

p_data = len(X) - split2

df['Pred_Signal'] = 0
df.iloc[-p_data:, df.columns.get_loc('Pred_Signal')] = clf.predict(X[split2:])

print(df['Pred_Signal'][-p_data:])

df['str_ret'] = df['Pred_Signal']*df['Return'].shift(-1)

df['strategy_cu_return'] = 0.0
df['market_cu_return'] = 0.0
df.iloc[-p_data:, df.columns.get_loc('strategy_cu_return')] = np.nancumsum(df['str_ret'][-p_data:])
df.iloc[-p_data:, df.columns.get_loc('market_cu_return')] = np.nancumsum(df['Return'][-p_data:])

Sharpe = (df['strategy_cu_return'][-1] - df['market_cu_return'][-1]) / np.nanstd(df['strategy_cu_return'][-p_data:])

plt.plot(df['strategy_cu_return'][-p_data:], color = 'g', label = 'Strategy Returns')
plt.plot(df['market_cu_return'][-p_data:], color = 'r', label = 'Market Returns')
plt.figtext(0.14, 0.9, s =  'Sharpe ration: {0:.2f}'.format(Sharpe))
plt.legend(loc = 'best')
plt.show()