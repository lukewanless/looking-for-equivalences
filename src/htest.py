import os
import pandas as pd
import numpy as np
from lr.text_processing.util import pre_process_nli_df
from lr.training.util import get_ternary_label
from lr.training.util import filter_df_by_label
from lr.text_processing.transformations.structural import entailment_internalization
from lr.stats.h_testing import DGP, get_matched_results, get_paired_t_statistic
from lr.stats.h_testing import get_boot_sample_under_H0, get_boot_p_value
from lr.stats.h_testing import LIMts_test
from lr.training.language_representation import Tfidf
from lr.models.logistic_regression import LRWrapper
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


train_path = "data/snli/train.csv"
dev_path = "data/snli/dev.csv"
result_path = "results/snli_lr_Tfidf_ent_int_1_1.csv"

debug = True
rho = 0.55
M = 10
E = 1
S = 1000
max_features = None

if debug:
    result_path = "results/t.csv"
    max_features = 500

train = pd.read_csv(train_path)
dev = pd.read_csv(dev_path)

if debug:
    train = train.head(100)
    dev = dev.head(100)

train = filter_df_by_label(train.dropna()).reset_index(drop=True)
dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
pre_process_nli_df(train)
pre_process_nli_df(dev)


hyperparams = {"RepresentationFunction": Tfidf,
               "max_features": max_features,
               "label_translation": get_ternary_label,
               "penalty": "l2",
               "C": 1,
               'solver': 'lbfgs'}

tests_results = LIMts_test(train=train,
                           dev=dev,
                           transformation=entailment_internalization,
                           rho=rho,
                           Model=LRWrapper,
                           hyperparams=hyperparams,
                           M=M,
                           E=E,
                           S=S,
                           verbose=True)

tests_results.to_csv(result_path, index=False)
