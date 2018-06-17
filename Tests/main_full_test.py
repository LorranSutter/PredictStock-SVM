import os
import sys
import time
import json
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
db_dir_res = 'db/results'
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

def indicators(stock):
    print("  Applying Indicators")
    t_indicators = time.time()
    stock.applyIndicators(ind_funcs_params, verbose = False)
    t_indicators = time.time() - t_indicators
    print("    Indicators applied")
    print("                      Time elapsed: {}".format(t_indicators))

    return t_indicators

def extraTrees(stock):
    print("  Applying Extra Trees CLF")
    t_extraTrees = time.time()
    stock.applyExtraTreesClassifier(nxt_day_predict)
    t_extraTrees = time.time() - t_extraTrees
    print("    Extra Tres CLF applied")
    print("                    Time elapsed: {}".format(t_extraTrees))

    return t_extraTrees

def ksvmeans(stock, random_state_kmeans, random_state_clf):
    print("  Fitting K-SVMeans")
    t_kSVMeans = time.time()
    stock.fit_kSVMeans(num_clusters = 4, 
                       classifier = 'OneVsOne',
                       random_state_kmeans = random_state_kmeans,
                       random_state_clf = random_state_clf,
                       consistent_clusters_kmeans = False,
                       consistent_clusters_multiclass = False,
                       extraTreesClf = True,
                       predictNext_k_day = nxt_day_predict,
                       extraTreesFirst = 0.9,
                       verbose = False)
    t_kSVMeans = time.time() - t_kSVMeans
    print("    K-SVMeans fitted")
    print("                    Time elapsed: {}".format(t_kSVMeans))

    return t_kSVMeans

def fit(stock, C_range, gamma_range, k_fold_num):
    print("  Fitting SVMs")
    t_fit = time.time()
    stock.fit(predictNext_k_day = nxt_day_predict,
              fit_type = None,
              maxRunTime = 25,
              parameters = {'C' : C_range, 'gamma' : gamma_range},
              n_jobs = 2,
              k_fold_num = k_fold_num,
              verbose = True)
    t_fit = time.time() - t_fit
    print("    SVMs fitted")
    print("                       Time elapsed: {}".format(t_fit))

    return t_fit

_gridSearch_ = True
_train_test_data_ = True

nxt_day_predict_list = [3,5,7,10]
extraTreesFirst_list = np.arange(0.1,0.31,0.01)
classifier_list = [None, 'OneVsOne']
num_k_fold_list = [3,5,10]
C_range = [2e-5*100**k for k in range(9)] # Max 2e11
gamma_range = [2e-15*100**k for k in range(8)] # Max 2e-1
rdms_kmeans = []
rdms_clf = []

if __name__ == "__main__":
    ticker = 'TSLA2'
    res_file = '{0}/{1}result.json'.format(db_dir_res, ticker)
    with open(res_file, 'w') as f:
        pass

    stock = Stock(ticker, considerOHL = False, train_test_data = _train_test_data_, train_size = 0.8)

    t_indicators = indicators(stock)
    t_extraTrees = extraTrees(stock)

    rdm_states = np.random.choice(len(stock.df.index), size = 30, replace = False)

    file_writting = dict()

    for rdm_state in rdm_states:
        res_preds_comp = ''
        t = ''

        # ! Problema
        # ! Ao refazer a extraTrees, o self.df está com os features já filtrados
        # ! Então ao fazer self.features = self.df[self.indicators_list]... alguns features não são encontrados

        # t_extraTrees = extraTrees(stock)

        # ! Problema
        # ! Ao refazer o fit do K-SVMeans, a coluna predict_k_days some. I don't know why

        t_kSVMeans = ksvmeans(stock, random_state_kmeans = rdm_state, random_state_clf = rdm_state)
        t_fit = fit(stock, C_range = C_range, gamma_range = gamma_range, k_fold_num = 3)

        labels_test = stock.predict_SVM_Cluster(stock.test)
        res_preds_comp = trainScore(stock, labels_test)

        print("Total time elapsed: {}".format(sum([t_indicators, t_extraTrees, t_kSVMeans, t_fit])))

        t = [t_indicators, t_extraTrees, t_kSVMeans, t_fit]

        file_writting['ticker'] = ticker
        file_writting['random_state'] = str(rdm_state)
        file_writting['time'] = t
        file_writting['preds_comp'] = res_preds_comp

        with open(res_file,'a') as f:
            json.dump(file_writting, f)
            f.write('\n')