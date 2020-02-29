from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import sys
import inspect
import unittest


currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.lr.text_processing.util import pre_process_nli_df  # noqa
from src.lr.text_processing.util import get_corpus  # noqa
from src.lr.training.util import get_ternary_label, filter_df_by_label  # noqa
from src.lr.training.language_representation import Tfidf  # noqa
from src.lr.training.util import get_ternary_label  # noqa
from src.lr.models.xgb import XGBCWrapper  # noqa


train_path = parentdir + "/src/data/toy/train.csv"
dev_path = parentdir + "/src/data/toy/dev.csv"


class BasicXGBTraining(unittest.TestCase):

    def test_xgb_training(self):
        train = pd.read_csv(train_path)
        dev = pd.read_csv(dev_path)
        train = filter_df_by_label(train.dropna()).reset_index(drop=True)
        dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
        pre_process_nli_df(train)
        pre_process_nli_df(dev)

        param_grid = {'n_estimators': range(10, 30, 5),
                      'max_depth': range(2, 31),
                      "reg_alpha": np.arange(0.05, 1.05, 0.05),
                      "reg_gamma": np.arange(0.05, 1.05, 0.05),
                      "learning_rate": np.arange(0.05, 1.05, 0.05),
                      "subsample": np.arange(0.05, 1.05, 0.05),
                      "colsample_bytree": np.arange(0.05, 1.05, 0.05)}

        hyperparams = {"RepresentationFunction": Tfidf,
                       "cv": 3,
                       "random_state": 123,
                       "verbose": False,
                       "n_jobs": 1,
                       "n_iter": 2,
                       "max_features": None,
                       "label_translation": get_ternary_label,
                       "param_grid": param_grid}

        repr_ = Tfidf(hyperparams)
        train_corpus = get_corpus(train)
        repr_.fit(train_corpus)
        X = repr_.transform(train_corpus)
        y = get_ternary_label(train)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=123)

        xg_cls = xgb.XGBClassifier(objective="multi:softprob",
                                   n_estimatores=10,
                                   random_state=123)

        gbm = xgb.XGBClassifier(objective="multi:softprob")
        params = hyperparams["param_grid"]

        randomized_cv = RandomizedSearchCV(param_distributions=hyperparams["param_grid"],
                                           estimator=gbm,
                                           scoring="accuracy",
                                           n_iter=hyperparams["n_iter"],
                                           cv=hyperparams["cv"],
                                           verbose=hyperparams["verbose"],
                                           random_state=hyperparams["random_state"])

        xg_cls.fit(X_train, y_train)
        y_pred = xg_cls.predict(X_test)
        untuned_acc = np.mean(y_pred == y_test)

        randomized_cv.fit(X, y)
        y_pred = randomized_cv.predict(X_test)
        tuned_acc = np.mean(y_pred == y_test)

        ratio = untuned_acc / tuned_acc

        hyperparams['random_state'] = 123
        model = XGBCWrapper(hyperparams)
        model.fit(train)

        self.assertTrue(model.model.best_score_ == randomized_cv.best_score_)
        self.assertTrue(model.model.best_params_ == randomized_cv.best_params_)
        self.assertTrue(ratio > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
