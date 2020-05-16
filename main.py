import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import Indicators as ind
from Stock import Stock

num_clusters = 5
nxt_day_predict = 7
db_dir = 'db'
extraRandomTree = True

ind_dict = {
    'SMA': ind.SMA,     # (df, n)
    'EMA': ind.EMA,     # (df, n)
    'MOM': ind.MOM,     # (df, n)
    'ROC': ind.ROC,     # (df, n)
    'ATR': ind.ATR,     # (df, n)
    'BBANDS': ind.BBANDS,  # (df, n, multiplier, middle)
    'PPSR': ind.PPSR,    # (df)
    'PPSRFIBO': ind.PPSRFIBO,  # (df)
    'STOK': ind.STOK,    # (df)
    'STO': ind.STO,     # (df, n)
    'TRIX': ind.TRIX,    # (df, n)
    'ADX': ind.ADX,     # (df, n, n_ADX)
    'MACD': ind.MACD,    # (df, n_fast, n_slow)
    'MASS': ind.MASS,   # (df)
    'VORTEX': ind.VORTEX,  # (df, n)
    'KST': ind.KST,     # (df, r1, r2, r3, r4, n1, n2, n3, n4)
    'RSI': ind.RSI,     # (df, n)
    'TSI': ind.TSI,     # (df, r, s)
    'ACCDIST': ind.ACCDIST,  # (df, n)
    'CHAIKIN': ind.CHAIKIN,  # (df)
    'MFI': ind.MFI,     # (df, n)
    'OBV': ind.OBV,     # (df, n)
    'FORCE': ind.FORCE,   # (df, n)
    'EOM': ind.EOM,     # (df, n)
    'CCI': ind.CCI,     # (df, n)
    'COPP': ind.COPP,    # (df, n)
    'KELCH': ind.KELCH,   # (df, n)
    'DONCH': ind.DONCH,   # (df, n)
    'ULTOSC': ind.ULTOSC   # (df)
}

if extraRandomTree:
    ind_funcs_params = []
    with open('db/FeaturesTest.txt', 'r') as f:
        for line in f:
            line = line.split(',')
            if len(line) == 1:
                ind_funcs_params.append([ind_dict[line[0][:-1]], None])
            else:
                params = line[1].split()
                params = map(int, params)
                ind_funcs_params.append([ind_dict[line[0]], tuple(params)])


def gridSearchEstimators(stock):
    for stockSVM in stock.stockSVMs:
        if stockSVM.clf is not None:
            print("Best estimators: C = {0} gamma = {1}"
                  .format(stockSVM.clf.best_estimator_.C, stockSVM.clf.best_estimator_.gamma))


def trainScore(stock, labels_test):
    preds = []
    for k, lab in enumerate(labels_test):
        preds.append(int(stock.predict_SVM(lab, stock.test[k:k+1])))

    res_preds_comp = [k == w for k, w in zip(stock.test_pred, preds)]

    preds2 = preds.copy()
    test_pred2 = stock.test_pred.copy()
    l = len(preds)

    print("{0} days : {1:.5f}%".format(0, sum(res_preds_comp)/l))
    for d in range(1, nxt_day_predict+3):
        preds.append(preds.pop(0))
        res_preds_comp = [k == w for k, w in zip(stock.test_pred, preds)]
        print("{0} days : {1:.5f}%".format(d, sum(res_preds_comp)/l))
    print()

    res_preds_comp = [k == w for k, w in zip(test_pred2, preds2)]
    print("{0} days : {1:.5f}%".format(0, sum(res_preds_comp)/l))
    for d in range(1, nxt_day_predict+3):
        preds2.pop(0)
        test_pred2.pop(-1)
        l = len(test_pred2)
        res_preds_comp = [k == w for k, w in zip(test_pred2, preds2)]
        print("{0} days : {1:.5f}%".format(d, sum(res_preds_comp)/l))
    print()


def plotStock(stock, _gridSearch_, _train_test_data_, labels_test=None):
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1, sharex=ax1)

    ax1.scatter(range(len(stock.df.index)),
                stock.df['Close'], c=stock.df['labels_kmeans'])
    ax2.scatter(range(len(stock.df.index)),
                stock.df['Close'], c=stock.df['labels'])
    ax3.scatter(range(len(stock.df.index)),
                stock.df['Close'], c=stock.df['labels'])
    if _gridSearch_ and _train_test_data_:
        ax3.scatter(range(len(stock.df.index), len(stock.df.index) +
                          len(stock.test.index)), stock.test['Close'], c=labels_test)
    plt.show()


_gridSearch_ = True
_train_test_data_ = True

C_range = [2e-5*100**k for k in range(5)]
gamma_range = [2e-15*100**k for k in range(5)]

if __name__ == "__main__":
    ticker = 'ZTS'

    stock = Stock(ticker, considerOHL=False,
                  train_test_data=_train_test_data_, train_size=0.8)

    stock.applyIndicators(ind_funcs_params)
    stock.applyExtraTreesClassifier(nxt_day_predict)
    stock.fit_kSVMeans(num_clusters=4,
                       classifier=None,
                       random_state_kmeans=None,
                       random_state_clf=None,
                       consistent_clusters_kmeans=False,
                       consistent_clusters_multiclass=False,
                       extraTreesClf=True,
                       predictNext_k_day=nxt_day_predict,
                       extraTreesFirst=1,
                       verbose=True)

    print("Init fit")
    t = time.time()
    stock.fit(predictNext_k_day=nxt_day_predict,
              fit_type='gridsearch',
              C=2e3,
              gamma=2e-15,
              parameters={'C': np.array(C_range), 'gamma': np.array(gamma_range)}, k_fold_num=3)
    print(time.time() - t)

    if _gridSearch_:
        gridSearchEstimators(stock)
        print()

    labels_test = None
    if _train_test_data_:
        labels_test = stock.predict_SVM_Cluster(stock.test)
        trainScore(stock, labels_test)

    plotStock(stock, _gridSearch_, _train_test_data_, labels_test)
