import numpy as np
import pandas as pd
import os
import sys
import inspect
import unittest
from joblib import load


currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.lr.text_processing.util import pre_process_nli_df  # noqa
from src.lr.training.util import get_binary_label, get_ternary_label  # noqa
from src.lr.models.logistic_regression import LRWrapper  # noqa
from src.lr.training.language_representation import BOW, Tfidf  # noqa


data_path = parentdir + "/src/data/toy/train.csv"
assert os.path.exists(data_path)


class BasicLrTraining(unittest.TestCase):
    @classmethod
    def setUp(cls):
        df = pd.read_csv(data_path)
        pre_process_nli_df(df)
        cls.df = df
        cls.path = "test.joblib"

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.path):
            os.remove(cls.path)

    def test_train_binary_BOW(self):
        hyperparams = {"RepresentationFunction": BOW,
                       "num_words": 500,
                       "label_translation": get_binary_label,
                       "penalty": "l2",
                       "C": 1,
                       'solver': 'lbfgs'}
        lr = LRWrapper(hyperparams)
        lr.fit(self.df)
        acc = lr.get_acc(self.df)
        baseline = (
            (self.df.label.value_counts() /
             self.df.shape[0])[
                1:]).sum()
        msg = "acc = {:.2f} baseline  = {:.2f}".format(acc, baseline)
        self.assertTrue(acc > baseline, msg)

    def test_train_ternary_Tfidf(self):
        hyperparams = {"RepresentationFunction": Tfidf,
                       "max_features": 500,
                       "label_translation": get_ternary_label,
                       "penalty": "l2",
                       "C": 1,
                       'solver': 'lbfgs'}
        lr = LRWrapper(hyperparams)
        lr.fit(self.df)
        acc = lr.get_acc(self.df)
        baseline = 0.4
        msg = "acc = {:.2f} baseline  = {:.2f}".format(acc, baseline)
        self.assertTrue(acc > baseline, msg)

    def test_save(self):
        hyperparams = {"RepresentationFunction": Tfidf,
                       "max_features": 500,
                       "label_translation": get_ternary_label,
                       "penalty": "l2",
                       "C": 1,
                       'solver': 'lbfgs'}
        lr = LRWrapper(hyperparams)
        lr.fit(self.df)
        scores = lr.get_acc(self.df)
        lr.save(self.path)
        del lr
        lr = load(self.path)
        new_scores = lr.get_acc(self.df)
        msg = "problem to save results"
        self.assertTrue(scores == new_scores, msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
