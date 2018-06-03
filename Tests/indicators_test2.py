import numpy as np
import pandas as pd
import Indicators as ind
import Indicators_original as indo

n = 10
n_fast = 5
n_slow = 10
n_ADX = 5
r1, r2, r3, r4, n1, n2, n3, n4 = 2, 3, 4, 5, 5, 10, 15, 20
r, s = 10, 5
db_dir = 'db'
ticker = 'TSLA2016'
df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True)
df2 = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True)

ind.MFI(df, n)
a2 = indo.MFI(df2, n)

keys = ['MFI_' + str(n)]

for n1, key in enumerate(keys):
    for n, k, w in zip(range(len(a2)), df[key], a2[key].values):
        if np.isnan(k) and np.isnan(w):
            continue
        if k != w:
            print(n, k, w)