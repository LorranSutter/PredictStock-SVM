import quandl
import datetime as dt
import pandas as pd
import pandas_datareader as web

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

from stockstats import StockDataFrame as sdf

style.use('ggplot')

db_dir = 'db'

start = dt.datetime(2013,6,1)
end = dt.datetime.now()
ticker = 'TSLA2016'
# ticker = 'TSLA'
# ticker = 'TSLA2013'
# ticker = 'BCHUSD'

# print("Getting {0} data...".format(ticker))
# df = web.DataReader(ticker,'morningstar', start, end)
# print("Data got!")

# df.to_csv(db_dir + '/{0}.csv'.format(ticker))

df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True, index_col = 0)

# df = quandl.get('BCHARTS/KRAKENUSD', returns ="pandas")

# df.reset_index(inplace = True)
# df.set_index('Date', inplace = True)
# df = df.drop('Symbol',axis = 1)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# values = df["Close"].values
# values = np.array([[k,values[k]] for k in range(len(values))])
#
# kmeans = KMeans(n_clusters=2)
# kmeans = kmeans.fit(values)
# labels = kmeans.predict(values)
#
# C = kmeans.cluster_centers_

# plt.scatter(C[:,0],C[:,1],marker='*')
# plt.scatter(values[:,0],values[:,1], c = labels)
# plt.show()

# from sklearn import svm
#
# clf = svm.SVC()
# clf.fit(values,labels)

# Open, High, Low, Close
# df_ohlc = df['Close'].resample('3D').ohlc()
# df_volume = df['Volume'].resample('3D').sum()

# df_ohlc = df_ohlc.reset_index()

# df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

# fig = plt.figure()
# ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
# ax1.xaxis_date()

# candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')

# ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)

# plt.show()


df['5ma'] = df['Close'].rolling(window = 5, min_periods = 0).mean()
df['8ma'] = df['Close'].rolling(window = 8, min_periods = 0).mean()
df['13ma'] = df['Close'].rolling(window = 13, min_periods = 0).mean()
df['13std'] = pd.stats.moments.rolling_std(df['Close'],13)
df['Upper Band'] = df['13ma'] + (df['13std']*2.2)
df['Lower Band'] = df['13ma'] -  (df['13std']*2.2)

stock = sdf.retype(df)

# f.loc['2013-06-01':]

MAs = df[['5ma','8ma','13ma']]
MAs_std = MAs.std(axis = 1)
MAs_std_good = MAs_std.where(MAs_std < 0.5).dropna()

if True:
    # ax1 = plt.subplot2grid((6,1),(0,0), rowspan=6, colspan=1)
    ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex=ax1)

    # Depending on data could be only df.index
    ax1.plot(df.index, df['close'])
    ax1.plot(df.index, df['5ma'], 'b')
    ax1.plot(df.index, df['8ma'], 'g')
    ax1.plot(df.index, df['13ma'], 'r')
    # ax1.plot(df.index, df['Upper Band'], 'm')
    # ax1.plot(df.index, df['Lower Band'], 'm')
    for k in range(len(MAs_std_good)):
        # ax1.axvline(MAs_std_good.index[k])
        ax1.plot(MAs_std_good.index[k], df.loc[MAs_std_good.index[k].strftime("%Y-%m-%d"),'close'], 'bo')

    if True:
        # ax2.bar(df.index, df['volume'])
        ax2.plot(df.index, stock['rsi_6'].values)
        ax2.axhline(70)
        ax2.axhline(30)

plt.show()


# import datetime

# import utils as ut

# import pandas as pd
# from matplotlib.finance import candlestick

# toString = lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S')

# records = ut.get_records("BCH","tickers","2017-12-21")

# records_last, records_date = [], []
# for rec in records:
#     records_last.append(float(rec['last']))
#     records_date.append(toString(rec['date']))

# tickerSeries = pd.Series(records_last, index = records_date)
