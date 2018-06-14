import sys
import numpy as np
import itertools as it
import multiprocessing

from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV

class StockSVM:

    def __init__(self, values):
        self.values = values
        self.predict_next_k_days = dict()
        self.clf = None
    
    def __repr__(self):
        return self.values.__repr__()

    def __str__(self):
        return self.values.__str__()

    def addPredictNext_K_Days(self, k_days, predict_next_k_day):
        self.predict_next_k_days[k_days] = predict_next_k_day

    def getValidFitParam(self, predictNext_k_day):
        vals, preds = [], []
        for val, pred in zip(self.values.values, self.predict_next_k_days[predictNext_k_day]):
            if not np.isnan(pred):
                vals.append(val)
                preds.append(int(pred))
        return vals, preds

    def fit(self, predictNext_k_day, C = 1.0, gamma = 'auto'):
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(predictNext_k_day))
            sys.exit()
        
        vals, preds = self.getValidFitParam(predictNext_k_day)

        svc = svm.SVC(C = C, gamma = gamma)
        self.clf = svc.fit(vals, preds)

    def fit_GridSearch(self, predictNext_k_day, parameters, n_jobs = 3, k_fold_num = None, verbose = 1):
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(predictNext_k_day))
            sys.exit()
        
        vals, preds = self.getValidFitParam(predictNext_k_day)

        svc = svm.SVC()
        self.clf = GridSearchCV(svc, parameters, n_jobs = n_jobs, cv = k_fold_num, verbose = verbose)
        self.clf = self.clf.fit(vals, preds)
    
    def fit_KFold(self, predictNext_k_day, parameters, n_jobs = 3, k_fold_num = None, verbose = 1):
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(predictNext_k_day))
            sys.exit()
        
        vals, preds = self.getValidFitParam(predictNext_k_day)

        # create all combinations
        # for k in it.product()

        # Use a string as a keyword argument
        # svm.SVC(**{'C': 1.0})

        # use multiprocessing

        svc = svm.SVC()
        kf = KFold(n_splits = k_fold_num)
        for train, test in kf.split(vals, preds):
            pass
        # self.clf = GridSearchCV(svc, parameters, n_jobs = n_jobs, cv = k_fold_num, verbose = verbose)
        # self.clf = self.clf.fit(vals, preds)

    def predict(self, X):
        return self.clf.predict(X)