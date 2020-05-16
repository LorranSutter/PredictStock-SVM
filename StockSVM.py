import sys
import time
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
        '''
        Add list of prediction according day key
        '''
        self.predict_next_k_days[k_days] = predict_next_k_day

    def getValidFitParam(self, predictNext_k_day):
        '''
        Return parameters what is not NaN
        '''
        vals, preds = [], []
        for val, pred in zip(self.values.values, self.predict_next_k_days[predictNext_k_day]):
            if not np.isnan(pred):
                vals.append(val)
                preds.append(int(pred))
        return vals, preds

    def fit(self, predictNext_k_day, C=1.0, gamma='auto'):
        '''
        Ordinary SVC fitting
        '''
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(
                predictNext_k_day))
            sys.exit()

        vals, preds = self.getValidFitParam(predictNext_k_day)

        svc = svm.SVC(C=C, gamma=gamma)
        self.clf = svc.fit(vals, preds)

    def fit_GridSearch(self, predictNext_k_day, parameters, n_jobs=3, k_fold_num=None, verbose=1):
        '''
        Grid Search Cross Validation Fitting
        '''
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(
                predictNext_k_day))
            sys.exit()

        vals, preds = self.getValidFitParam(predictNext_k_day)

        svc = svm.SVC()
        self.clf = GridSearchCV(
            svc, parameters, n_jobs=n_jobs, cv=k_fold_num, verbose=verbose)
        self.clf = self.clf.fit(vals, preds)

    def fit_Cross_Validation(self, predictNext_k_day, parameters, k_fold_num=3, maxRunTime=25, verbose=1):
        if predictNext_k_day not in self.predict_next_k_days.keys():
            print('{0} day(s) for predictNext not defined! Please, add a vector for that days first!'.format(
                predictNext_k_day))
            sys.exit()

        vals, preds = self.getValidFitParam(predictNext_k_day)
        keys = parameters.keys()

        def __fitSVC__(svc, X, y, queue):
            clf = queue.get()
            clf = svc.fit(X, y)
            queue.put(clf)

        def __runProcess__(svc, X_train, y_train, queue):
            try:
                p = multiprocessing.Process(target=__fitSVC__,
                                            name="fitSVC",
                                            args=(svc, X_train, y_train, queue))
                t = 0
                interrupted = False
                p.start()  # Start fitting process
                while p.is_alive():
                    print("Time elapsed: {0:.2f}".format(t), end='\r')
                    # Reach maximum time, terminate process
                    if t >= maxRunTime:
                        p.terminate()
                        p.join()
                        interrupted = True
                        break
                    time.sleep(0.01)
                    t += 0.01
            except Exception:
                p.terminate()
                p.join()

            return interrupted

        best_score = 0
        best_svc = None
        best_estimators = None
        queue = multiprocessing.Queue()
        for params in it.product(*parameters.values()):

            if verbose != 0:
                print("Params: ", params)

            # Create model with current params
            dict_params = dict(zip(keys, params))
            svc = svm.SVC(**dict_params)

            score_sum = 0
            num_scores = 0
            kf = KFold(n_splits=k_fold_num)
            # Split vals, preds in k_fold_num folds
            for train_ids, test_ids in kf.split(vals, preds):

                # Get vals, preds train data according current fold
                X_train = [vals[k] for k in train_ids]
                y_train = [preds[k] for k in train_ids]

                last_clf = None
                queue.put(last_clf)  # Share last_clf variable

                # Init fitting parallel process
                print(" Init process")
                interrupted = __runProcess__(svc, X_train, y_train, queue)

                # If process have finished, not interrupted by max run time
                if not interrupted:
                    last_clf = queue.get()  # Get shared last_clf variable

                    # Get vals, preds test data according current fold
                    X_test = [vals[k] for k in test_ids]
                    y_test = [preds[k] for k in test_ids]

                    # Add score for current fold
                    score = last_clf.score(X_test, y_test)
                    score_sum += score
                    num_scores += 1
                    print(" Last score: ", score)
                    print(" Score sum: ", score_sum)
                    print()
                else:
                    print(" Process interrupted!")
                    print()

            if num_scores != 0:
                avarege_score = score_sum / num_scores
                print(" Average score: ", avarege_score)
                print()
                if avarege_score > best_score:
                    best_score = avarege_score
                    best_svc = svc
                    best_estimators = params

        print()
        print("Best Score: ", best_score)
        print("Best SVC: ", best_svc)
        print("Best Estimators: ", best_estimators)
        print()

        if best_svc is not None:
            self.clf = best_svc.fit(vals, preds)

    # * Prediction function
    def predict(self, X):
        return self.clf.predict(X)
