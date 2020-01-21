from joblib import dump
import os
import sys
import inspect
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from text_processing.util import get_corpus  # noqa


class LRWrapper():
    """
    logistic regression wrapper
    """

    def __init__(self, hyperparams):
        """
        :param labels: list of labels
        :type labels: [str]
        """
        self.RepresentationFunction = hyperparams["RepresentationFunction"]
        self.label_translation = hyperparams["label_translation"]

        if "param_grid" not in hyperparams:
            self.penalty = hyperparams["penalty"]
            self.C = hyperparams["C"]
            self.solver = hyperparams['solver']
            self.model = LogisticRegression(penalty=self.penalty,
                                            C=self.C,
                                            solver=self.solver)
        else:
            param_grid = hyperparams["param_grid"]
            cv = hyperparams["cv"]
            verbose = hyperparams["verbose"] 
            logistic = LogisticRegression()
            self.model = GridSearchCV(logistic,
                                      param_grid=param_grid,
                                      cv=cv,
                                      verbose=verbose,
                                      n_jobs=-1)
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
        x = self.transform(df)
        y = self.label_translation(df)
        self.model = self.model.fit(x, y)

    def predict(self, df):
        x = self.transform(df)
        return self.model.predict(x)

    def get_score(self, df, k=1):
        x = self.transform(df)
        return self.model.predict_proba(x)[:, k]

    def get_acc(self, df):
        x = self.predict(df)
        y = self.label_translation(df)
        return np.mean(x == y)

    def save(self, path):
        _ = dump(self, path)
