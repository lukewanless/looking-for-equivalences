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

from src.lr.text_processing.transformations.structural import entailment_internalization  # noqa
from src.lr.training.util import get_ternary_label, filter_df_by_label  # noqa
from src.lr.text_processing.util import pre_process_nli_df  # noqa
from src.lr.training.language_representation import Tfidf  # noqa
from src.lr.training.util import get_ternary_label  # noqa
from src.lr.stats.h_testing import DGP  # noqa
from src.lr.models.logistic_regression import LRWrapper  # noqa
from src.lr.stats.h_testing import Majority  # noqa
from src.lr.stats.h_testing import get_matched_results, get_paired_t_statistic  # noqa
from src.lr.stats.h_testing import get_boot_sample_under_H0  # noqa
from src.lr.stats.h_testing import get_boot_p_value  # noqa
from src.lr.stats.h_testing import LIMts_test  # noqa

train_path = parentdir + "/src/data/toy/train.csv"
dev_path = parentdir + "/src/data/toy/dev.csv"


class Htesting(unittest.TestCase):

    def test_random_seed(self):
        train_trans = entailment_internalization
        dev_trans = entailment_internalization
        train = pd.read_csv(train_path)
        dev = pd.read_csv(dev_path)
        train = filter_df_by_label(train.dropna()).reset_index(drop=True)
        dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
        pre_process_nli_df(train)
        pre_process_nli_df(dev)
        dev_t = dev_trans(dev)

        np.random.seed(1234)

        param_grid = {"C": np.linspace(0, 3, 50),
                      "penalty": ["l2"]}

        hyperparams = {"RepresentationFunction": Tfidf,
                       "cv": 3,
                       "solver": 'lbfgs',
                       "random_state": None,
                       "verbose": False,
                       "n_jobs": 1,
                       "n_iter": 2,
                       "max_features": None,
                       "label_translation": get_ternary_label,
                       "param_grid": param_grid}

        dgp = DGP(train, train_trans, rho=0.3)
        train_ = dgp.sample()

        E = 2
        models = []
        for e in range(E):
            lr = LRWrapper(hyperparams)
            lr.fit(train_)
            models.append(lr)

        lr = Majority(models)
        results = get_matched_results(dev,
                                      dev_t,
                                      lr,
                                      lr.label_translation)
        t_obs = get_paired_t_statistic(results)

        S = 1000
        t_boots = []
        for _ in range(S):
            boot_sample_result = get_boot_sample_under_H0(results)
            boot_t = get_paired_t_statistic(boot_sample_result)
            t_boots.append(boot_t)

        t_boots = pd.Series(t_boots)
        p_value = get_boot_p_value(t_boots, t_obs)

        train = pd.read_csv(train_path)
        dev = pd.read_csv(dev_path)
        train = filter_df_by_label(train.dropna()).reset_index(drop=True)
        dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)

        pre_process_nli_df(train)
        pre_process_nli_df(dev)

        combined_information = LIMts_test(train=train,
                                          dev=dev,
                                          train_transformation=train_trans,
                                          dev_transformation=dev_trans,
                                          rho=0.3,
                                          Model=LRWrapper,
                                          hyperparams=hyperparams,
                                          M=1,
                                          E=2,
                                          S=1000,
                                          verbose=False,
                                          random_state=1234)

        self.assertTrue(
            np.all(
                combined_information.validation_accuracy == results.A.mean()))
        self.assertTrue(
            np.all(
                combined_information.transformed_validation_accuracy == results.B.mean()))
        self.assertTrue(
            np.all(
                combined_information.observable_t_stats == t_obs))
        self.assertTrue(np.all(combined_information.p_value == p_value))

        boot_c = [
            c for c in combined_information.columns if c.find("boot") > -1]
        new_boot = combined_information[boot_c].transpose()
        self.assertTrue(np.all(new_boot.sum() == t_boots.sum()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
