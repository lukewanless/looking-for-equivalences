import os
import sys
import inspect
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from text_processing.util import get_corpus  # noqa


class XGBCWrapper():
    """
    XGBClassifier wrapper
    """

    def __init__(self, hyperparams):
        """
        :param labels: list of labels
        :type labels: [str]
        """
        self.RepresentationFunction = hyperparams["RepresentationFunction"]
        self.label_translation = hyperparams["label_translation"]

        if "param_grid" not in hyperparams:
            self.n_estimators = hyperparams['n_estimators']
            self.max_depth = hyperparams["max_depth"]
            self.reg_alpha = hyperparams["reg_alpha"]
            self.reg_gamma = hyperparams["reg_gamma"]
            self.learning_rate = hyperparams["learning_rate"]
            self.subsample = hyperparams["subsample"]
            self.colsample_bytree = hyperparams["colsample_bytree"]

            self.model = xgb.XGBClassifier(objective="multi:softprob",
                                           n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           reg_alpha=self.reg_alpha,
                                           reg_gamma=self.reg_gamma,
                                           learning_rate=self.learning_rate,
                                           subsample=self.subsample,
                                           colsample_bytree=self.colsample_bytree)
        else:
            param_grid = hyperparams["param_grid"]
            cv = hyperparams["cv"]
            verbose = hyperparams["verbose"]
            n_jobs = hyperparams["n_jobs"]
            n_iter = hyperparams["n_iter"]
            random_state = hyperparams["random_state"]

            gbm = xgb.XGBClassifier(objective="multi:softprob")
            self.model = RandomizedSearchCV(gbm,
                                            param_distributions=param_grid,
                                            cv=cv,
                                            n_iter=n_iter,
                                            random_state=random_state,
                                            verbose=verbose,
                                            n_jobs=n_jobs)
        self.repr = self.RepresentationFunction(hyperparams=hyperparams)
        self.repr_fit = False

    def fit_representation(self, df):
        corpus = get_corpus(df)
        self.repr.fit(corpus)
        self.repr_fit = True

    def transform(self, df):
        corpus = get_corpus(df)
        return self.repr.transform(corpus)

    def fit(self, df):
        if not self.repr_fit:
            self.fit_representation(df)
        X = self.transform(df)
        y = self.label_translation(df)
        self.model = self.model.fit(X, y)

    def predict(self, df):
        x = self.transform(df)
        return self.model.predict(x)

    def get_score(self):
        return self.model.best_score_

    def get_acc(self, df):
        x = self.predict(df)
        y = self.label_translation(df)
        return np.mean(x == y)

    def save(self, path):
        _ = dump(self, path)
