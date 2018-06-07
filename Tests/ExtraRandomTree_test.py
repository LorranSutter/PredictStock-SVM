import numpy as np
import pandas as pd

import Indicators as ind

from Stock import Stock

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

ind_dict = {
             'SMA' : ind.SMA,     # (df, n)
             'EMA' : ind.EMA,     # (df, n)
             'MOM' : ind.MOM,     # (df, n)
             'ROC' : ind.ROC,     # (df, n)
             'ATR' : ind.ATR,     # (df, n)
             'BBANDS' : ind.BBANDS,  # (df, n, multiplier, middle)
             'PPSR' : ind.PPSR,    # (df)
             'PPSRFIBO' : ind.PPSRFIBO,# (df)
             'STOK' : ind.STOK,    # (df)
             'STO' : ind.STO,     # (df, n)
             'TRIX' : ind.TRIX,    # (df, n)
             'ADX' : ind.ADX,     # (df, n, n_ADX)
             'MACD' : ind.MACD,    # (df, n_fast, n_slow)
             'MASS' : ind.MASS,   # (df)
             'VORTEX' : ind.VORTEX,  # (df, n)
             'KST' : ind.KST,     # (df, r1, r2, r3, r4, n1, n2, n3, n4)
             'RSI' : ind.RSI,     # (df, n)
             'TSI' : ind.TSI,     # (df, r, s)
             'ACCDIST' : ind.ACCDIST, # (df, n)
             'CHAIKIN' : ind.CHAIKIN, # (df)
             'MFI' : ind.MFI,     # (df, n)
             'OBV' : ind.OBV,     # (df, n)
             'FORCE' : ind.FORCE,   # (df, n)
             'EOM' : ind.EOM,     # (df, n)
             'CCI' : ind.CCI,     # (df, n)
             'COPP' : ind.COPP,    # (df, n)
             'KELCH' : ind.KELCH,   # (df, n)
             'DONCH' : ind.DONCH,   # (df, n)
             'ULTOSC' : ind.ULTOSC   # (df)
             }

ind_funcs = []
ind_params = []
with open('db/FeaturesTest.txt', 'r') as f:
    for line in f:
        line = line.split(',')
        if len(line) == 1:
            ind_funcs.append([True, ind_dict[line[0][:-1]]])
            ind_params.append(None)
        else:
            ind_funcs.append([True, ind_dict[line[0]]])
            params = line[1].split()
            params = map(int, params)
            ind_params.append(tuple(params))

db_dir = 'db'
ticker = 'TSLA2'
days_predict = 7

if 'TSLA' in ticker:
    df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True)
else:
    df = pd.read_csv(db_dir + '/stocks/{0}/{1}.csv'.format(ticker[0].upper(), ticker), parse_dates = True)

stock = Stock(ticker, considerOHL = False, train_test_data = False)

stock.applyIndicators(ind_funcs, ind_params)

stock.applyPredict(days_predict)
stock.removeNaN()

X = stock.df.drop(['Close','High','Low','Open','Volume','predict_' + str(days_predict) + '_days'], axis = 1)
Y = stock.df['predict_' + str(days_predict) + '_days'].tolist()

for d in range(days_predict+1):
    Y.append(Y.pop(0))

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

func_importance = {k : w for k,w in zip(X.columns, model.feature_importances_)}

func_imp_sort = sorted(func_importance.items(), key = lambda x: x[1], reverse=True)

split_size = int(len(func_imp_sort))
first_func_imp_sort = func_imp_sort[:split_size]

# Have problems with names like MACD_Signal and BB_Up, '_' into the name
with open('db/FeaturesTestOut.txt','w') as f:
    for func_imp in first_func_imp_sort:
        func_imp = func_imp[0].split('_')
        func_name = ''
        for letter in func_imp[0]:
            if letter.islower():
                break
            func_name += letter
        func = func_name + ',' + ' ' + ' '.join(func_imp[1:])
        f.write(func + '\n')

# stock.df.isnull().sum().values
# for k in stock.df.columns:
#     vals = stock.df[k]
#     for i, v in enumerate(vals.values):
#         if v == np.inf or v == -np.inf:
#             print(k, vals.index[i], v)