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
n_ADX = 5
r1, r2, r3, r4, n1, n2, n3, n4 = 2, 3, 4, 5, 5, 10, 15, 20
r, s = 10, 5
num_clusters = 5
nxt_day_predict = 7
db_dir = 'db'

ind_funcs = [
             [False, ind.MA],      # (df, n)
             [True , ind.EMA],     # (df, n)
             [True , ind.MOM],     # (df, n)
             [False, ind.ROC],     # (df, n)
             [True , ind.ATR],     # (df, n)
             [True , ind.BBANDS],  # (df, n, multiplier, middle)
             [False, ind.PPSR],    # (df)
             [False, ind.PPSRFIBO],# (df)
             [False, ind.STOK],    # (df)
             [False, ind.STO],     # (df, n)
             [False, ind.TRIX],    # (df, n)
             [False, ind.ADX],     # (df, n, n_ADX)
             [False, ind.MACD],    # (df, n_fast, n_slow)
             [False, ind.MassI],   # (df)
             [False, ind.Vortex],  # (df, n)
             [False, ind.KST],     # (df, r1, r2, r3, r4, n1, n2, n3, n4)
             [True , ind.RSI],     # (df, n)
             [False, ind.TSI],     # (df, r, s)
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

_gridSearch_ = True

if __name__ == "__main__":
    ticker = 'TSLA2'

    stock = Stock(ticker, considerOHL = False, train_test_data = True, train_size = 0.8)

    stock.applyIndicators(ind_funcs, ind_params)
    print()
    stock.fit_kSVMeans(num_clusters = 4,\
                       random_state_kmeans = None,\
                       random_state_clf = 40,\
                       classifier = None,\
                       consistent_clusters = True)
    print('split')
    stock.splitByLabel2()
    print('split end')

    # for stockSVM in stock.stockSVMs:
    #     print(len(stockSVM.values))

    # ind.MA(df,3)
    stock.applyPredict(nxt_day_predict)

    # stock.fit(predictNext_k_day = nxt_day_predict,
    #           gridSearch = _gridSearch_, 
    #           parameters = {'C' : np.linspace(2e-5,2e3,10), 'gamma' : np.linspace(2e-15,2e3,10)}, k_fold_num = 5)
    stock.fit(predictNext_k_day = nxt_day_predict,
              gridSearch = _gridSearch_, 
              parameters = {'C' : np.linspace(2e-5,2e3,10), 'gamma' : [2e-15]}, k_fold_num = 3)
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
    for d in range(1,nxt_day_predict+10):
        preds.append(preds.pop(0))
        res_preds_comp = [k == w for k,w in zip(stock.test_pred, preds)]
        print("{0} days : {1:.5f}%".format(d, sum(res_preds_comp)/l))
    print()

    # stock.removeNaN()

    # pred = df['predict_3_days'].values
    # df = df.drop(['predict_3_days'], axis = 1).values

    # parameters = {'C': np.linspace(10e-16,10e-10,100), 'gamma': np.linspace(10e-16,10e-10,100)}

    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters, verbose=1, n_jobs=3)
    # clf = clf.fit(df, pred)

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