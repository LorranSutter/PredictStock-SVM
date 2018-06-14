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

def gridSearchEstimators(stock):
    for stockSVM in stock.stockSVMs:
        if stockSVM.clf is not None:
            print("Best estimators: C = {0} gamma = {1}"\
                .format(stockSVM.clf.best_estimator_.C, stockSVM.clf.best_estimator_.gamma))

def trainScore(stock, labels_test, verbose = False):
    preds = []
    for k, lab in enumerate(labels_test):
        preds.append(int(stock.predict_SVM(lab, [stock.test.iloc[k]])))
        # print(lab, stock.predict_SVM(lab, stock.test[k:k+1]))
    
    test_pred = stock.test_pred.copy()
    l = len(preds)

    res_preds_comp = []

    res_preds_comp.append(sum([k == w for k,w in zip(test_pred, preds)])/l)
    if verbose:
        print("{0} days : {1:.5f}%".format(0, res_preds_comp[-1]))
    for d in range(1,nxt_day_predict+3):
        preds.pop(0)
        test_pred.pop(-1)
        l = len(test_pred)
        res_preds_comp.append(sum([k == w for k,w in zip(test_pred, preds)])/l)
        if verbose:
            print("{0} days : {1:.5f}%".format(d, res_preds_comp[-1]))
    if verbose:
        print()
    
    return res_preds_comp

_gridSearch_ = True
_train_test_data_ = True

nxt_day_predict_list = [1,3,5,7,10]
extraTreesFirst_list = np.arange(0.1,0.31,0.01)
classifier_list = [None, 'OneVsOne']
num_k_fold_list = [3,5,10]
C_range = [2e-5*100**k for k in range(11)]
gamma_range = [2e-15*100**k for k in range(10)]

if __name__ == "__main__":
    ticker = 'TSLA2'

    t = time.time()

    stock = Stock(ticker, considerOHL = False, train_test_data = _train_test_data_, train_size = 0.8)

    print("Calculating indicators...")
    stock.applyIndicators(ind_funcs_params, verbose = False)
    print("Indicators calculated!\n")

    stock.applyExtraTreesClassifier(nxt_day_predict)
    stock.fit_kSVMeans(num_clusters = 4, 
                       classifier = 'OneVsOne',
                       random_state_kmeans = None,
                       random_state_clf = None,
                       consistent_clusters_kmeans = True,
                       consistent_clusters_multiclass = True,
                       extraTreesClf = True,
                       predictNext_k_day = nxt_day_predict,
                       extraTreesFirst = 0.2,
                       verbose = True)

    print()
    stock.fit(predictNext_k_day = nxt_day_predict,
              gridSearch = _gridSearch_, 
              parameters = {'C' : np.linspace(2e-5,2e3,30), 'gamma' : [2e-15]}, n_jobs = 2, k_fold_num = 5)

    print(time.time() - t)
    # res_preds_comp = []
    # k = 0
    # for c in np.linspace(2e-5,2e3,30):
    #     for g in np.linspace(2e-15,2e3,30):
    #         stock.fit(predictNext_k_day = nxt_day_predict, C = c, gamma = g)
    #         labels_test = stock.predict_SVM_Cluster(stock.test)
    #         res_preds_comp.append(trainScore(stock, labels_test))
    #         print("Iteration " + str(k), end = "\r")
    #         k += 1
    
    if False:
        if _gridSearch_:
            print('grid Estimators 1\n')
            gridSearchEstimators(stock)
            print()

        labels_test1 = None        
        if _train_test_data_:
            print('Score test 1\n')
            labels_test1 = stock.predict_SVM_Cluster(stock.test)
            trainScore(stock, labels_test1)