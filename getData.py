import os
import sys
# import quandl
import datetime as dt
import pandas as pd
import pandas_datareader as web

db_dir = 'db'
stocks = 'stocks'

start = dt.datetime(1950,1,1)
end = dt.datetime.now()

if os.path.exists('db/lastId.txt'):
    with open('db/lastId.txt', 'r') as f:
        id_begin = f.readline()
else:
    with open('lastId.txt', 'w') as f:
        f.write('-1')
        id_begin = -1

nasdaq = pd.read_csv(db_dir + '/NASDAQ.csv')
nasdaq = nasdaq.iloc[id_begin:]
symbols = nasdaq['Symbol'].values

for k, symbol in enumerate(symbols):
    if '$' in symbol:
        continue
    if not os.path.exists('{0}/{1}/{2}/{3}.csv'.format(db_dir, stocks, symbol[0].upper(), symbol)):
        print("Getting {0} {1} data...".format(id_begin + k, symbol))
        df = web.DataReader(symbol,'morningstar', start, end)
        print("Data got!")

        path = db_dir + '/' + stocks + '/' + symbol[0].upper()
        if not os.path.exists(path):
            os.mkdir(path)
        df.to_csv(path + '/{0}.csv'.format(symbol))

    with open('db/lastId.txt', 'w') as f:
        f.write(str(id_begin + k + 1))

with open('db/lastId.txt', 'w') as f:
    f.write('-1')

# df = quandl.get('BCHARTS/KRAKENUSD', returns ="pandas")

# df.reset_index(inplace = True)
# df.set_index('Date', inplace = True)
# df = df.drop('Symbol',axis = 1)
