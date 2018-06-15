import os
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

def getGoodTickers(parameter_list):
    goodTickers = []
    badTickers = []
    if os.path.isfile(db_dir + '/badTickers.txt'):
        with open(db_dir + '/badTickers.txt','r') as f:
            badTickers = [line.replace('\n','') for line in f]
    if os.path.isfile(db_dir + '/goodTickers.txt'):
        with open(db_dir + '/goodTickers.txt','r') as f:
            goodTickers = [line.replace('\n','') for line in f]
    
    tickers = pickle.load(open(db_dir + "/sp500tickers.pickle",'rb'))
    tickers = [t for t in tickers if t not in badTickers and t not in goodTickers]

    exceptions = dict()

    for ticker in tickers:
        try:

            print("Trying {} ticker".format(ticker))

            stock = Stock(ticker, considerOHL = False, train_test_data = _train_test_data_, train_size = 0.8)

            stock.applyIndicators(ind_funcs_params, verbose = False)

            stock.applyExtraTreesClassifier(nxt_day_predict)
            stock.fit_kSVMeans(num_clusters = 4, 
                            classifier = 'OneVsOne',
                            random_state_kmeans = None,
                            random_state_clf = None,
                            consistent_clusters_multiclass = True,
                            extraTreesClf = True,
                            predictNext_k_day = nxt_day_predict,
                            extraTreesFirst = 0.2,
                            verbose = False)

            print()
            stock.fit(predictNext_k_day = nxt_day_predict,
                    fit_type = 'girdsearch', 
                    parameters = {'C' : np.linspace(2e-5,2e3,30), 'gamma' : [2e-15]}, n_jobs = 2, k_fold_num = 3)
        
            print("\n{} ticker SUCESS!\n".format(ticker))

            goodTickers.append(ticker)
            with open(db_dir + '/goodTickers.txt','w') as f:
                for t in goodTickers:
                    f.write(t)
                    f.write('\n')

        except Exception as e:
            print("\n{} ticker FAIL!\n".format(ticker))
            exceptions[ticker] = e
            badTickers.append(ticker)

            with open(db_dir + '/badTickers.txt','w') as f:
                for t in badTickers:
                    f.write(t)
                    f.write('\n')

            pass


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
C_range = [2e-5*100**k for k in range(11)]
C_range = [2e-3,2e2]
gamma_range = [2e-15*100**k for k in range(10)]
gamma_range = [2e-15]

if __name__ == "__main__":

    goodTickers = []
    if os.path.isfile(db_dir + '/goodShortTickers.txt'):
        with open(db_dir + '/goodShortTickers.txt','r') as f:
            goodTickers = [line.replace('\n','') for line in f]
    else:
        print("\nWhere are the goodTickers file, man?\n")
        sys.exit()

    goodTickers = ['TSLA2']

    for ticker in goodTickers:
        try:

            # nxt = False
            # if os.path.isfile(db_dir + '/tickersTimeResult.txt'):
            #     with open(db_dir + '/tickersTimeResult.txt','a') as f:
            #         for line in f:
            #             if ticker in line:
            #                 nxt = True
            #                 break
            
            # if nxt: continue

            print("Trying {} ticker".format(ticker))

            stock = Stock(ticker, considerOHL = False, train_test_data = _train_test_data_, train_size = 0.8)
            
            print("  Applying Indicators")
            t_indicators = time.time()
            stock.applyIndicators(ind_funcs_params, verbose = False)
            t_indicators = time.time() - t_indicators
            print("    Indicators applied")
            print("                      Time elapsed: {}".format(t_indicators))

            print("  Applying Extra Trees Clf")
            t_extraTrees = time.time()
            stock.applyExtraTreesClassifier(nxt_day_predict)
            t_extraTrees = time.time() - t_extraTrees
            print("    Extra Trees Clf applied")
            print("                           Time elapsed: {}".format(t_extraTrees))

            print("  Fitting K-SVMeans")
            t_kSVMeans = time.time()
            stock.fit_kSVMeans(num_clusters = 4, 
                               classifier = None,
                               random_state_kmeans = None,
                               random_state_clf = None,
                               consistent_clusters_kmeans = True,
                               consistent_clusters_multiclass = True,
                               extraTreesClf = True,
                               predictNext_k_day = nxt_day_predict,
                               extraTreesFirst = 1,
                               verbose = False)
            t_kSVMeans = time.time() - t_kSVMeans
            print("    K-SVMeans fitted")
            print("                    Time elapsed: {}".format(t_kSVMeans))

            print("  Fitting Cross Validation")
            t_crossValidation = time.time()
            stock.fit(predictNext_k_day = nxt_day_predict,
                      fit_type = 'crossvalidation',
                      maxRunTime = 10,
                      parameters = {'C' : C_range, 'gamma' : gamma_range}, n_jobs = 2, k_fold_num = 3)
            t_crossValidation = time.time() - t_crossValidation
            print("    GridSearchCV fitted")
            print("                       Time elapsed: {}".format(t_crossValidation))
        
            labels_test1 = stock.predict_SVM_Cluster(stock.test)
            res_preds_comp = trainScore(stock, labels_test1)

            print("Total time elapsed: {}".format(sum([t_indicators, t_extraTrees, t_kSVMeans, t_crossValidation])))

            print("{} ticker SUCESS!\n".format(ticker))

            res_preds_comp = ','.join(list(map(str,res_preds_comp)))
            t = ','.join(list(map(str,[t_indicators, t_extraTrees, t_kSVMeans, t_crossValidation])))
            with open(db_dir + '/tickersTimeResult.txt','a') as f:
                f.write("{0};{1};{2}\n".format(ticker, t, res_preds_comp))

        except Exception as e:
            print("\n{} ticker FAIL!\n".format(ticker))
            print(e)
            # with open(db_dir + '/tickersTimeResult.txt','a') as f:
            #     f.write(ticker + ' FAIL!\n')
            pass
