import numpy as np
import pandas as pd
import os
import sys
import inspect
import unittest
import shutil
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

    @classmethod
    def tearDown(cls):
        if os.path.exists("my_version"):
            shutil.rmtree("my_version")

    def test_random_seed(self):

        if not os.path.exists("my_version"):
            os.mkdir("my_version")

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
                       "param_grid": param_grid,
                       "random_state_list": [11, 2],
                       "dgp_seed_list": [1234, 12],
                       "data_set_name": "toy",
                       "transformation_name": "entailment internalization",
                       "rho": 0.3,
                       "model_name_or_path": "logistic regression",
                       "number_of_samples": 2,
                       "number_of_models": 1,
                       "number_of_simulations": 1000,
                       "verbose": False,
                       "output_dir": "my_version"}

        train = pd.read_csv(train_path)
        dev = pd.read_csv(dev_path)
        train = filter_df_by_label(train.dropna()).reset_index(drop=True)
        dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
        pre_process_nli_df(train)
        pre_process_nli_df(dev)
        train_trans = entailment_internalization
        dev_trans = entailment_internalization

        combined_information = LIMts_test(train=train,
                                          dev=dev,
                                          train_transformation=train_trans,
                                          dev_transformation=dev_trans,
                                          Model=LRWrapper,
                                          hyperparams=hyperparams)

        self.assertTrue(combined_information.observable_t_stats[0] == -4.264014327112209)
        self.assertTrue(combined_information.observable_t_stats[1] == -5.097911392420601)


if __name__ == '__main__':
    unittest.main(verbosity=2)
