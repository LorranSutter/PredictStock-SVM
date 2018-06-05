import os
import sys
import numpy as np
import pandas as pd

from collections import Counter

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from StockSVM import StockSVM

class Stock:

    def __init__(self, ticker, considerOHL = False, remVolZero = True, train_test_data = False, train_size = 0.8):
        self.__considerOHL__ = considerOHL
        self.__db_dir__ = 'db/stocks'
        self.__OHL__  = []
        self.__default_columns__ = ['Date','Close','High','Low','Open','Volume']

        if not considerOHL:
            self.__OHL__ = ['Open','High','Low']

        self.__readTicker__(ticker)
        self.df.set_index('Date', inplace = True)

        self.__delExtraColumns__()

        if 'Volume' in self.df.columns and remVolZero:
            self.df = self.df[self.df['Volume'] != 0]
        
        self.__default_main_columns__ = self.df.columns.tolist()

        self.train_test_data = False
        if train_test_data:
            if train_size <= 0 or train_size >= 1:
                print('train_size param must be in (0,1) interval!')
                sys.exit()
            self.train_test_data = True
            self.train_size = train_size
            self.test = None
            self.test_pred = []
        
        self.indicators_list = []
        self.clf = None
        self.stockSVMs = []
        self.available_labels = []
    
    def __repr__(self):
        return self.df.__repr__()

    def __str__(self):
        return self.df.__str__()

    # * Read data stock from file
    def __readTicker__(self, ticker):
        if 'TSLA' in ticker:
            self.df = pd.read_csv('db' + '/{0}.csv'.format(ticker), parse_dates = True)
        else:
            try:
                self.df = pd.read_csv(self.__db_dir__ + '/{0}/{1}.csv'.format(ticker[0].upper(), ticker), parse_dates = True)
            except FileNotFoundError as e:
                print(e)
                sys.exit()

    # * Delete extra columns
    def __delExtraColumns__(self):
        for col in [col for col in self.df.columns if col not in self.__default_columns__]:
            self.df.pop(col)
            print(col + " extra column removed!")

    # * Add predict_next_k_day array to respective stockSVM
    def __add_PredictNxtDay_to_SVM__(self, k_days):
        if self.stockSVMs != []:
            for label in range(max(self.df['labels']) + 1):
                predict_k_days_col = self.df['predict_' + str(k_days) + '_days']
                predict_k_days_col = predict_k_days_col[self.df['labels'] == label].values
                self.stockSVMs[label].addPredictNext_K_Days(k_days, predict_k_days_col)

    def __set_available_labels__(self):
        for k in self.df['labels']:
            if k not in self.available_labels:
                self.available_labels.append(k)

    # * Split data in train and test data
    def __train_test_split__(self):     
        split = int(self.train_size * len(self.df))
        self.df, self.test = self.df[:split], self.df[split:]

        allowed_list = self.__default_main_columns__ + self.indicators_list
        if not self.__considerOHL__:
            default = set(self.__OHL__)
            allowed_list = [col for col in allowed_list if col not in default]
        self.test = self.test[allowed_list]

    # * Remove rows which has at least one NA/NaN/NULL value
    def removeNaN(self):
        self.df.dropna(inplace = True)

    # * Apply indicators
    def applyIndicators(self, ind_funcs, ind_params, replaceInf = True, remNaN = True):
        for f, p in zip(ind_funcs, ind_params):

            # Calculate if inidicator is True
            if f[0]:
                if p != None:
                    f[1](self.df, *p)
                else:
                    f[1](self.df)
                print(f[1].__name__ + " indicator calculated!")
        
        default = set(self.__default_main_columns__)
        self.indicators_list = [col for col in self.df.columns if col not in default]

        # Replace infinity values by NaN
        if replaceInf:
            self.df.replace([np.inf, -np.inf], np.nan, inplace = True)
        
        # Remove rows which has at least one NA/NaN/NULL value 
        if remNaN:
            self.df.dropna(inplace = True)
        
        if self.train_test_data:
            self.__train_test_split__()
    
    # * Say if all clusters has at least num_clusters items
    def __consistent_clusters__(self, num_clusters, labels):
        freqs = Counter(labels).values()
        for freq in Counter(labels).values():
            print(freq)
            if freq <= num_clusters:
                return False
        if min(freqs) / max(freqs) < 0.02:
            return False
        return True

    # * K-Means algorithm
    def __fit_KMeans__(self, num_clusters = 5, random_state_kmeans = None):
        self.clf = KMeans(n_clusters = num_clusters, init = 'k-means++', algorithm = 'full', n_jobs = -1, random_state = random_state_kmeans)
        self.clf = self.clf.fit(self.df.values)
        
        return self.clf.predict(self.df.values)

    def __fit_Multiclass_Classifier__(self, classifier, random_state_clf, labels):
        if classifier == 'OneVsOne':
            self.clf = OneVsOneClassifier(
                                            svm.LinearSVC(random_state = random_state_clf))\
                                            .fit(self.df.drop(['labels_kmeans'], axis = 1).values, labels)
        elif classifier == 'OneVsRest':
            self.clf = OneVsRestClassifier(
                                            svm.LinearSVC(random_state = random_state_clf))\
                                            .fit(self.df.drop(['labels_kmeans'], axis = 1).values, labels)
        else:
            print('Invalid input classifier!')
            sys.exit()
        
        return self.clf.predict(self.df.drop(['labels_kmeans'], axis = 1).values)

    # ! Improve
    # * K-SVMeans clustering
    def fit_kSVMeans(self, num_clusters = 5, classifier = 'OneVsOne', random_state_kmeans = None, random_state_clf = None, consistent_clusters = False):
        for k in self.__OHL__:
            if k in self.df.columns:
                self.df = self.df.drop(k, axis = 1)
        
        labels = self.__fit_KMeans__(num_clusters = num_clusters, random_state_kmeans = random_state_kmeans)

        if consistent_clusters:
            print('KMeans')
            while not self.__consistent_clusters__(num_clusters, labels):
                labels = self.__fit_KMeans__(num_clusters = num_clusters, random_state_kmeans = None)
                print()

        self.df['labels_kmeans'] = labels

        if classifier is not None:
            labels_clf = self.__fit_Multiclass_Classifier__(classifier = classifier, random_state_clf = random_state_clf, labels = labels)
            if consistent_clusters:
                print('Multiclass')
                while not self.__consistent_clusters__(max(labels_clf) + 1, labels_clf):
                    labels_clf = self.__fit_Multiclass_Classifier__(classifier = classifier, random_state_clf = None, labels = labels)
                    print()
        
        if classifier is not None:
            self.df['labels'] = labels_clf
        else:
            self.df['labels'] = labels

        self.__set_available_labels__()

        # return labels

    # * Apply predict next n days
    def applyPredict(self, k_days = 1):
        if k_days < 1:
            print("k_days must be greater than 0!")
            sys.exit()

        prices = self.df['Close'].values
        predict_next = [np.nan for k in range(len(self.df.index))]

        if self.train_test_data:
            prices = np.append(prices, self.test['Close'].values)
            predict_next.extend([np.nan for k in range(len(self.test.index))])

        if k_days == 1:
            for k in range(len(prices) - 1):
                if prices[k+1] > prices[k]:
                    predict_next[k] = 1
                elif prices[k+1] < prices[k]:
                    predict_next[k] = -1
                else:
                    predict_next[k] = 0
        else:
            if self.train_test_data:
                sma = self.df['Close'].append(self.test['Close'])
                sma = sma.rolling(k_days).mean().values
            else:
                sma = self.df['Close'].rolling(k_days).mean().values
            for k in range(k_days - 1, len(sma) - k_days):
                if sma[k + k_days] > sma[k]:
                    predict_next[k] = 1
                elif sma[k + k_days] < sma[k]:
                    predict_next[k] = -1
                else:
                    predict_next[k] = 0
        
        if self.train_test_data:
            self.df['predict_' + str(k_days) + '_days'] = predict_next[:len(self.df.index)]
            self.test_pred = predict_next[len(self.df.index):]
        else:
            self.df['predict_' + str(k_days) + '_days'] = predict_next

        self.__add_PredictNxtDay_to_SVM__(k_days)

        # return predict_next
    
    # * Split and return an array
    # TODO Remove Open, High, Low
    def splitByLabel1(self):
        if self.clf is None:
            print('Data must be clusterized before split by label!')
            sys.exit()

        predict_to_drop = [col for col in self.df.columns if 'predict' in col]
        if predict_to_drop is not None:
            values = self.df.drop(predict_to_drop, axis = 1).values
        else:
            values = self.df.values
        
        a = [[] for k in range(max(self.df['labels'])+1)]
        for row, label in zip(values, self.df['labels']):
            a[label].append(row)
        
        self.stockSVMs = [StockSVM(new_stockSVM) for new_stockSVM in a]
        # self.stockSVMs = [StockSVM(new_stockSVM) for new_stockSVM in filter(None, a)]

    # * Split and return a data frame
    # TODO Remove Open, High, Low
    def splitByLabel2(self):
        if self.clf is None:
            print('Data must be clusterized before split by label!')
            sys.exit()
        
        self.stockSVMs = []

        for label in range(max(self.df['labels'])+1):
            new_stockSVM = self.df[self.df['labels'] == label]
            # if new_stockSVM.empty:
            #     continue

            new_stockSVM = new_stockSVM.drop('labels', axis = 1)

            for k in self.__OHL__ + ['labels_kmeans', 'labels']:
                if k in new_stockSVM.columns:
                    new_stockSVM = new_stockSVM.drop(k, axis = 1)

            for k in new_stockSVM.columns:
                if 'predict' in k:
                    new_stockSVM = new_stockSVM.drop(k, axis = 1)

            new_stockSVM = StockSVM(new_stockSVM)
            self.stockSVMs.append(new_stockSVM)
    
    # * Fit data in each SVM Cluster
    def fit(self, predictNext_k_day, C = 1.0, gamma = 'auto', gridSearch = False, parameters = None, n_jobs = 3, k_fold_num = None, verbose = 1):
        if self.stockSVMs != []:
            for svm in self.stockSVMs:
                if not svm.values.empty:
                    if gridSearch:
                        svm.fitGridSearch(predictNext_k_day = predictNext_k_day,
                                          parameters = parameters,
                                          n_jobs = n_jobs,
                                          k_fold_num = k_fold_num,
                                          verbose = verbose)
                    else:
                        svm.fit(predictNext_k_day, C, gamma)
    
    # * Predict which SVM Cluster df param belongs
    def predict_SVM_Cluster(self, df):
        return self.clf.predict(df)

    def predict_SVM(self, cluster_id, df):
        return self.stockSVMs[cluster_id].predict(df)