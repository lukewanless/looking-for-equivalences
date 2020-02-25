from lr.stats.h_testing import LIMts_test
from lr.stats.h_testing import get_boot_p_value
from lr.stats.h_testing import get_boot_sample_under_H0
from lr.stats.h_testing import get_matched_results, get_paired_t_statistic
from lr.stats.h_testing import Majority
from lr.models.logistic_regression import LRWrapper
from lr.stats.h_testing import DGP
from lr.training.util import get_ternary_label
from lr.training.language_representation import Tfidf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lr.text_processing.util import pre_process_nli_df
from lr.training.util import get_ternary_label, filter_df_by_label
from lr.text_processing.transformations.structural import entailment_internalization
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


train_path = "data/toy/train.csv"
dev_path = "data/toy/dev.csv"


train_trans = entailment_internalization
dev_trans = entailment_internalization

train = pd.read_csv(train_path)
dev = pd.read_csv(dev_path)
train = filter_df_by_label(train.dropna()).reset_index(drop=True)
dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)

dev_t = dev_trans(dev)

pre_process_nli_df(train)
pre_process_nli_df(dev)

max_features = None

param_grid = {"C": np.linspace(0, 3, 50),
              "penalty": ["l2"]}


hyperparams = {"RepresentationFunction": Tfidf,
               "cv": 3,
               "solver": 'lbfgs',
               "random_state": None,
               "verbose": False,
               "n_jobs": 1,
               "n_iter": 2,
               "max_features": max_features,
               "label_translation": get_ternary_label,
               "param_grid": param_grid}

np.random.seed(1234)

dgp = DGP(train, train_trans, rho=0.3)

train_ = dgp.sample()

E = 2


models = []

for e in range(E):
    lr = LRWrapper(hyperparams)
    lr.fit(train_)
    models.append(lr)

accs = [lr.get_acc(train_) for lr in models]
lr = Majority(models)


results = get_matched_results(dev, dev_t, lr, lr.label_translation)
t_obs = get_paired_t_statistic(results)


S = 1000

t_boots = []

for _ in range(S):
    boot_sample_result = get_boot_sample_under_H0(results)
    boot_t = get_paired_t_statistic(boot_sample_result)
    t_boots.append(boot_t)

t_boots = pd.Series(t_boots)

p_value = get_boot_p_value(t_boots, t_obs)


b = LIMts_test(train=train,
               dev=dev,
               train_transformation=train_trans,
               dev_transformation=dev_trans,
               rho=0.3,
               Model=LRWrapper,
               hyperparams=hyperparams,
               M=1,
               E=2,
               S=1000,
               verbose=True,
               random_state=1234)



print(b.observable_t_stats, t_obs)
print(b.p_value , p_value)


# assert np.all(b.observable_t_stats == t_obs)
# assert np.all(b.p_value == p_value)
