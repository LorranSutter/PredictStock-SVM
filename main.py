import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# from mpl_toolkits.mplot3d import Axes3D

import Indicators as ind
from KSVMeans import KSVMeans

from Stock import Stock

n = 7
multiplier = 2.2
middle = False
n_fast = 12
n_slow = 26
n_ADX = 14
r1, r2, r3, r4, n1, n2, n3, n4 = 10, 15, 20, 30, 10, 10, 10, 15
r, s = 25, 13
num_clusters = 5
nxt_day_predict = 7
db_dir = 'db'
extraRandomTree = True

ind_funcs = [
             [False, ind.SMA],     # (df, n)
             [False, ind.EMA],     # (df, n)
             [True , ind.MOM],     # (df, n)
             [False, ind.ROC],     # (df, n)
             [False, ind.ATR],     # (df, n)
             [True , ind.BBANDS],  # (df, n, multiplier, middle)
             [False, ind.PPSR],    # (df)
             [False, ind.PPSRFIBO],# (df)
             [True , ind.STOK],    # (df)
             [False, ind.STO],     # (df, n)
             [False, ind.TRIX],    # (df, n)
             [False, ind.ADX],     # (df, n, n_ADX)
             [True , ind.MACD],    # (df, n_fast, n_slow)
             [False, ind.MassI],   # (df)
             [False, ind.Vortex],  # (df, n)
             [True , ind.KST],     # (df, r1, r2, r3, r4, n1, n2, n3, n4)
             [True , ind.RSI],     # (df, n)
             [True , ind.TSI],     # (df, r, s)
             [False, ind.ACCDIST], # (df, n)
             [False, ind.Chaikin], # (df)
             [True , ind.MFI],     # (df, n)
             [True , ind.OBV],     # (df, n)
             [False, ind.FORCE],   # (df, n)
             [False, ind.EOM],     # (df, n)
             [True , ind.CCI],     # (df, n)
             [False, ind.COPP],    # (df, n)
             [False, ind.KELCH],   # (df, n)
             [False, ind.ULTOSC],  # (df)
             [False, ind.DONCH]    # (df, n)
             ]

ind_params = [
              (n,),
              (n,),
              (n,),
              (n,),
              (n,),
              (n, multiplier, middle),
              None,
              None,
              None,
              (n,),
              (n,),
              (n, n_ADX),
              (n_fast, n_slow),
              None,
              (n,),
              (r1, r2, r3, r4, n1, n2, n3, n4),
              (n,),
              (r, s),
              (n,),
              None,
              (n,),
              (n,),
              (n,),
              (n,),
              (n,),
              (n,),
              (n,),
              None,
              (n,)
            ]

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
             'MassI' : ind.MassI,   # (df)
             'Vortex' : ind.Vortex,  # (df, n)
             'KST' : ind.KST,     # (df, r1, r2, r3, r4, n1, n2, n3, n4)
             'RSI' : ind.RSI,     # (df, n)
             'TSI' : ind.TSI,     # (df, r, s)
             'ACCDIST' : ind.ACCDIST, # (df, n)
             'Chaikin' : ind.Chaikin, # (df)
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
    ind_funcs = []
    ind_params = []
    with open('db/FeaturesTestOut.txt', 'r') as f:
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

_gridSearch_ = True

if __name__ == "__main__":
    ticker = 'ZTS'

    stock = Stock(ticker, considerOHL = False, train_test_data = True, train_size = 0.8)

    stock.applyIndicators(ind_funcs, ind_params)
    print()
    stock.fit_kSVMeans(num_clusters = 4,\
                       random_state_kmeans = None,\
                       random_state_clf = None,\
                       classifier = 'OneVsOne',\
                       consistent_clusters = True)
    print('split')
    stock.splitByLabel2()
    print('split end')

    stock.applyPredict(nxt_day_predict)

    # stock.fit(predictNext_k_day = nxt_day_predict,
    #           gridSearch = _gridSearch_, 
    #           parameters = {'C' : np.linspace(2e-5,2e3,20), 'gamma' : np.linspace(2e-15,2e3,5)}, k_fold_num = 5)
    stock.fit(predictNext_k_day = nxt_day_predict,
              gridSearch = _gridSearch_, 
              parameters = {'C' : np.linspace(2e-5,2e3,100), 'gamma' : [2e-15]}, k_fold_num = 5)
    print()

    if _gridSearch_:
        for stockSVM in stock.stockSVMs:
            if stockSVM.clf is not None:
                print("Best estimators: C = {0} gamma = {1}"\
                    .format(stockSVM.clf.best_estimator_.C,stockSVM.clf.best_estimator_.gamma))
        print()

    labels_test = stock.predict_SVM_Cluster(stock.test)

    preds = []
    for k, lab in enumerate(labels_test):
        preds.append(int(stock.predict_SVM(lab, stock.test[k:k+1])))
        # print(lab, stock.predict_SVM(lab, stock.test[k:k+1]))

    res_preds_comp = [k == w for k,w in zip(stock.test_pred, preds)]

    l = len(stock.test)
    print("{0} days : {1:.5f}%".format(0, sum(res_preds_comp)/l))
    for d in range(1,nxt_day_predict+3):
        preds.append(preds.pop(0))
        res_preds_comp = [k == w for k,w in zip(stock.test_pred, preds)]
        print("{0} days : {1:.5f}%".format(d, sum(res_preds_comp)/l))
    print()

    # ax1 = plt.subplot2grid((2,1),(0,0), rowspan=1, colspan=1)
    # ax2 = plt.subplot2grid((2,1),(1,0), rowspan=1, colspan=1)
    ax1 = plt.subplot2grid((3,1),(0,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((3,1),(1,0), rowspan=1, colspan=1, sharex = ax1)
    ax3 = plt.subplot2grid((3,1),(2,0), rowspan=1, colspan=1, sharex = ax1)

    # ax1.scatter(range(len(stock.df.index)), stock.df['Close'], c=labels1)
    # ax2.scatter(range(len(stock.df.index)), stock.df['Close'], c=labels)

    # ax1 = plt.subplot2grid((4,1),(0,0), rowspan=2, colspan=1)
    # ax2 = plt.subplot2grid((4,1),(2,0), rowspan=1, colspan=1)
    # ax3 = plt.subplot2grid((4,1),(3,0), rowspan=1, colspan=1)

    # df = df.dropna()

    # ax1.plot(df['Close'].values)
    # ax1.plot(df['MA_3'].values)
    # ax1.plot(df['MA_10'].values)
    # ax2.bar(df.index, df['predict_3_days'].values)
    # ax3.bar(df.index, df['predict_10_days'].values)

    ax1.scatter(range(len(stock.df.index)), stock.df['Close'], c = stock.df['labels_kmeans'])
    ax2.scatter(range(len(stock.df.index)), stock.df['Close'], c = stock.df['labels'])
    ax3.scatter(range(len(stock.df.index)), stock.df['Close'], c = stock.df['labels'])
    if _gridSearch_:
        ax3.scatter(range(len(stock.df.index), len(stock.df.index)+len(stock.test.index)), stock.test['Close'], c = labels_test)
    plt.show()