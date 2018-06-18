import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import Indicators as ind
from KSVMeans import KSVMeans

from Stock import Stock

num_clusters = 5
nxt_day_predict = 7
db_dir = 'db'
extraRandomTree = True

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

if extraRandomTree:
    ind_funcs_params = []
    with open('db/FeaturesTestOut2.txt', 'r') as f:
        for line in f:
            line = line.split(',')
            if len(line) == 1:
                ind_funcs_params.append([ind_dict[line[0][:-1]], None])
            else:
                params = line[1].split()
                params = map(int, params)
                ind_funcs_params.append([ind_dict[line[0]], tuple(params)])

_gridSearch_ = True
_train_test_data_ = True

if __name__ == "__main__":
    ticker = 'TSLA'

    stock = Stock(ticker, considerOHL = False, train_test_data = _train_test_data_, train_size = 0.8)
    stock.applyIndicators(ind_funcs_params, verbose = True)
    stock.applyPredict(nxt_day_predict)
    stock.fit_kSVMeans(num_clusters = 4,\
                       random_state_kmeans = 40,\
                       random_state_clf = 40,\
                       classifier = 'OneVsOne',\
                       consistent_clusters_multiclass = True)