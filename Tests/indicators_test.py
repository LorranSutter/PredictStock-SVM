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
ticker = 'TSLA'
df = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True)
df2 = pd.read_csv(db_dir + '/{0}.csv'.format(ticker), parse_dates = True)

funcs = [
        'MA',
        'EMA',
        'MOM',
        'ROC',
        'ATR',
        'BBANDS',
        'PPSR',
        'STOK',
        'STO,',
        'TRIX',
        'ADX',
        'MACD',
        'MassI',
        'Vortex',
        'KST',
        'RSI',
        'TSI',
        'ACCDIST',
        'Chaikin',
        'MFI',
        'OBV',
        'FORCE',
        'EOM',
        'CCI',
        'COPP',
        'KELCHd',
        'ULTOSC',
        'DONCH']

ind_funcs = [
             ind.MA(df, n),
             ind.EMA(df, n),
             ind.MOM(df, n),
             ind.ROC(df, n),
             ind.ATR(df, n),
             ind.BBANDS(df, n),
             ind.PPSR(df),
             ind.STOK(df),
             ind.STO(df, n),
             ind.TRIX(df, n),
             ind.ADX(df, n, n_ADX),
             ind.MACD(df, n_fast, n_slow),
             ind.MassI(df),
             ind.Vortex(df, n),
             ind.KST(df, r1, r2, r3, r4, n1, n2, n3, n4),
             ind.RSI(df, n),
             ind.TSI(df, r, s),
             ind.ACCDIST(df, n),
             ind.Chaikin(df),
             ind.MFI(df, n),
             ind.OBV(df, n),
             ind.FORCE(df, n),
             ind.EOM(df, n),
             ind.CCI(df, n),
             ind.COPP(df, n),
             ind.KELCH(df, n),
             ind.ULTOSC(df),
             ind.DONCH(df, n)
             ]

indo_funcs = [
              indo.MA(df2, n),
              indo.EMA(df2, n),
              indo.MOM(df2, n),
              indo.ROC(df2, n),
              indo.ATR(df2, n),
              indo.BBANDS(df2, n),
              indo.PPSR(df2),
              indo.STOK(df2),
              indo.STO(df2, n),
              indo.TRIX(df2, n),
              indo.ADX(df2, n, n_ADX),
              indo.MACD(df2, n_fast, n_slow),
              indo.MassI(df2),
              indo.Vortex(df2, n),
              indo.KST(df2, r1, r2, r3, r4, n1, n2, n3, n4),
              indo.RSI(df2, n),
              indo.TSI(df2, r, s),
              indo.ACCDIST(df2, n),
              indo.Chaikin(df2),
              indo.MFI(df2, n),
              indo.OBV(df2, n),
              indo.FORCE(df2, n),
              indo.EOM(df2, n),
              indo.CCI(df2, n),
              indo.COPP(df2, n),
              indo.KELCH(df2, n),
              indo.ULTOSC(df2),
              indo.DONCH(df2, n)
             ]

errs = {k : [] for k in funcs}

for func in indo_funcs:
    df2 = df2.join(func[func.columns.difference(['Symbol', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'])])
    
funcs1 = df[df.columns.difference(['Symbol', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'])]
funcs2 = df2[df2.columns.difference(['Symbol', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'])]

for k, func_name, array1, array2 in zip(range(len(funcs)), funcs, funcs1.columns, funcs2.columns):
    array1 = funcs1[array1]
    array2 = funcs2[array2]
    if np.isnan(array1) and np.isnan(array2):
        continue
    if np.isnan(array1) and not np.isnan(array2):
        errs[func_name].append([k, array1, array2])
    elif not np.isnan(array1) and np.isnan(array2):
        errs[func_name].append([k, array1, array2])
    elif array1 != array2:
        errs[func_name].append([k, array1, array2])

if False:
    for funcs_name, funcs2 in zip(funcs, indo_funcs):
        funcs2 = funcs2[funcs2.columns.difference(['Symbol', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'])]
        funcs2 = [funcs2[k].values for k in funcs2.columns]
        if len(funcs2) == 1:
            funcs2 = funcs2[0]

        for k, arrays1, arrays2 in zip(range(len(funcs2)), df[df.columns[7+i]], funcs2):
            if np.isnan(arrays1) and np.isnan(arrays2):
                continue
            if np.isnan(arrays1) and not np.isnan(arrays2):
                errs[funcs_name].append([k, arrays1, arrays2])
            elif not np.isnan(arrays1) and np.isnan(arrays2):
                errs[funcs_name].append([k, arrays1, arrays2])
            elif arrays1 != arrays2:
                errs[funcs_name].append([k, arrays1, arrays2])
            # else:
            #     for w, a1, a2 in zip(range(len(arrays1)), arrays1, arrays2):
            #         if np.isnan(a1) and np.isnan(a2):
            #             continue
            #         if np.isnan(a1) and not np.isnan(a2):
            #             errs[funcs_name].append([k, w, a1, a2])
            #         elif not np.isnan(a1) and np.isnan(a2):
            #             errs[funcs_name].append([k, w, a1, a2])
            #         elif a1 != a2:
            #             errs[funcs_name].append([k, w, a1, a2])

errs_diff = {k : errs[k] for k in errs.keys() if errs[k] != []}