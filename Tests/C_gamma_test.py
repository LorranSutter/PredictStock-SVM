import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import itertools as it
import multiprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

import Indicators as ind
from KSVMeans import KSVMeans

from Stock import Stock

num_clusters = 5
nxt_day_predict = 5
db_dir_res = 'db/results/C_gamma'

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
maxRunTime = 120
k_fold_num = 3

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
with open('db/FeaturesTest.txt', 'r') as f:
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
                       classifier = None,
                       random_state_kmeans = random_state_kmeans,
                       random_state_clf = random_state_clf,
                       consistent_clusters_kmeans = True,
                       consistent_clusters_multiclass = True,
                       extraTreesClf = True,
                       predictNext_k_day = nxt_day_predict,
                       extraTreesFirst = 0.2,
                       verbose = False)
    t_kSVMeans = time.time() - t_kSVMeans
    print("    K-SVMeans fitted")
    print("                    Time elapsed: {}".format(t_kSVMeans))

    return t_kSVMeans

def fit(queue_stock, k_fold_num, queue_time, C, gamma):
    print("  Fitting SVMs")
    stock = queue_stock.get()
    t_fit = queue_time.get()
    t_fit = time.time()
    stock.fit(predictNext_k_day = nxt_day_predict,
              fit_type = None,
              C = C,
              gamma = gamma,
              maxRunTime = 25,
              n_jobs = 2,
              k_fold_num = k_fold_num,
              verbose = False)
    t_fit = time.time() - t_fit
    queue_time.put(t_fit)
    queue_stock.put(stock)
    print("  SVMs fitted")
    print("                       Time elapsed: {}\n".format(t_fit))

def __runProcess__(queue_stock, svm_params, k_fold_num, maxRunTime, queue_time):
    try:
        p = multiprocessing.Process(target = fit,
                                    name = "fit",
                                    args = (queue_stock, k_fold_num, queue_time),
                                    kwargs = svm_params)
        t = 0
        interrupted = False
        p.start() # Start fitting process
        while p.is_alive():
            # print("Time elapsed: {0:.2f}".format(t))
            # sys.stdout.write('\x1b[1A') # Go to the above line
            # sys.stdout.write('\x1b[2K') # Erase line

            # Reach maximum time, terminate process
            if t >= maxRunTime:
                p.terminate()
                p.join(10)
                print("Interrupted!")
                interrupted = True
                break
            time.sleep(0.01)
            t += 0.01
    except Exception:
        p.terminate()
        p.join(10)
    
    return interrupted

def get_random_states(ticker):
    path = 'db/results/C_gamma/' + ticker + '/'
    if not os.path.isdir(path):
        os.mkdir(path)

    random_state_extraTrees = []
    random_state_kmeans = []
    random_state_clf = []

    files = os.listdir(path)
    for f_txt in files:
        with open(path + f_txt, 'r') as f:
            for line in f:
                line = json.loads(line)
                if 'random_state_extraTrees' in line.keys():
                    if line['random_state_extraTrees'] not in random_state_extraTrees:
                        random_state_extraTrees.append(line['random_state_extraTrees'])
                if 'random_state_kmeans' in line.keys():
                    if line['random_state_kmeans'] not in random_state_kmeans:
                        random_state_kmeans.append(line['random_state_kmeans'])
                if 'random_state_clf' in line.keys():
                    if line['random_state_clf'] not in random_state_clf:
                        random_state_clf.append(line['random_state_clf'])
    
    return random_state_extraTrees, random_state_kmeans, random_state_clf

def main(ticker, main_it):
    random_state_extraTrees, random_state_kmeans, random_state_clf = get_random_states(ticker)
    res_file = '{0}/{1}/{1}_{2}.json'.format(db_dir_res, ticker, int(time.time()))
    with open(res_file, 'w') as f:
        pass

    while True:
        stock = Stock(ticker, considerOHL = False, train_test_data = _train_test_data_, train_size = 0.8)

        t_indicators = indicators(stock)
        # ? Problema
        # ? Ao refazer em nova iteração a extraTrees, o self.df está com os features já filtrados
        # ? Então ao fazer self.features = self.df[self.indicators_list]... alguns features não são encontrados
        t_extraTrees = extraTrees(stock)
        t_kSVMeans = ksvmeans(stock, random_state_kmeans = None, random_state_clf = None)

        if stock.random_state_extraTrees in random_state_extraTrees:
            continue
        elif stock.random_state_kmeans in random_state_kmeans:
            continue
        elif stock.random_state_clf in random_state_clf:
            continue
        else:
            random_state_extraTrees.append(stock.random_state_extraTrees)
            random_state_kmeans.append(stock.random_state_kmeans)
            random_state_clf.append(stock.random_state_clf)
            break

    parameters = {'C' : C_range, 'gamma' : gamma_range}
    keys = parameters.keys()

    size = len(C_range) * len(gamma_range)

    file_writting = dict()

    manager = multiprocessing.Manager()
    queue_time = manager.Queue()
    queue_stock = manager.Queue()

    for k, params in enumerate(it.product(*parameters.values()), start = 1):

        svm_params = dict(zip(keys,params))
        file_writting = {'ticker' : ticker,
                         'predict_nxt_day' : nxt_day_predict,
                         'cluster_number' : max(stock.df['labels_kmeans'])-1,
                         'random_state_extraTrees' : stock.random_state_extraTrees,
                         'random_state_kmeans' : stock.random_state_kmeans,
                         'random_state_clf' : stock.random_state_clf,
                         **svm_params,
                         'time' : [],
                         'preds_comp' : []}
        
        try:
            print("\nMain it {0}/{1} - Sub it {2}/{3}".format(main_it[0],main_it[1],k,size))
            res_preds_comp = ''
            t = ''        

            t_fit = 0
            queue_time.put(t_fit)
            queue_stock.put(stock)
            interrupted = __runProcess__(queue_stock, svm_params, k_fold_num, maxRunTime, queue_time)

            if interrupted or queue_time.empty():
                file_writting['ERROR'] = 'interrupted'
            else:
                t_fit = queue_time.get()
                stock = queue_stock.get()

                labels_test = stock.predict_SVM_Cluster(stock.test)
                res_preds_comp = trainScore(stock, labels_test)

                t = [t_indicators, t_extraTrees, t_kSVMeans, t_fit]

                file_writting['time'] = t
                file_writting['preds_comp'] = res_preds_comp

            with open(res_file,'a') as f:
                json.dump(file_writting, f)
                f.write('\n')
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            file_writting['ERROR'] = str(e)
            with open(res_file,'a') as f:
                json.dump(file_writting, f)
                f.write('\n')

main_it = 8
if __name__ == "__main__":
    ticker = 'GOOGL'
    try:
        for k in range(1, main_it+1):
            main(ticker, [k,main_it])
    except KeyboardInterrupt:
        print("Keyboard Interruption")
    except Exception as e:
        print(e)