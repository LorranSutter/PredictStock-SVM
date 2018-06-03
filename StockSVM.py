import sys
import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV

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

    def fit(self, predictNext_k_day, C = 1.0, gamma = 'auto'):
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(predictNext_k_day))
            sys.exit()
        
        vals, predNxt_k_day = [], []
        for val, pred in zip(self.values.values, self.predict_next_k_days[predictNext_k_day]):
            if not np.isnan(pred):
                vals.append(val)
                predNxt_k_day.append(int(pred))

        svc = svm.SVC(C = C, gamma = gamma)
        self.clf = svc.fit(vals, predNxt_k_day)

    def fitGridSearch(self, predictNext_k_day, parameters, n_jobs = 3, k_fold_num = None, verbose = 1):
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(predictNext_k_day))
            sys.exit()
        
        vals, predNxt_k_day = [], []
        for val, pred in zip(self.values.values, self.predict_next_k_days[predictNext_k_day]):
            if not np.isnan(pred):
                vals.append(val)
                predNxt_k_day.append(pred)

        svc = svm.SVC()
        self.clf = GridSearchCV(svc, parameters, n_jobs = n_jobs, cv = k_fold_num, verbose = verbose)
        self.clf = self.clf.fit(vals, predNxt_k_day)
    
    def predict(self, X):
        return self.clf.predict(X)