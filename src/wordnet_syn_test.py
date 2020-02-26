import pandas as pd
import numpy as np
from lr.text_processing.util import pre_process_nli_df
from lr.training.util import get_ternary_label, filter_df_by_label
from lr.text_processing.transformations.wordnet import path_base_transformation
from lr.text_processing.transformations.wordnet import path_base_transformation_p
from lr.text_processing.transformations.wordnet import path_base_transformation_h
from lr.training.language_representation import Tfidf
from lr.models.logistic_regression import LRWrapper
from lr.stats.h_testing import LIMts_test
from IPython.display import display, HTML
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# params


def wordsyn_test(transformation_type, max_features, C_size, cv,
                 n_jobs, n_iter, rho, M, E, S,
                 verbose, random_state, debug):

    results_path = "results/snli/lr/wordnet_{}_rho_{}".format(
        transformation_type, rho)
    results_path = results_path.replace(".", "p")
    results_path = results_path + ".csv"

    # Loading data

    train_path = "data/snli/train.csv"
    dev_path = "data/snli/dev.csv"

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    train = filter_df_by_label(train.dropna()).reset_index(drop=True)
    dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)

    if debug:
        train = train.head(10000)
        dev = dev.head(1000)

    pre_process_nli_df(train)
    pre_process_nli_df(dev)

    # creating transformation function

    train_path_mod = "data/snli/train_p_h_syn_noun.csv"
    dev_path_mod = "data/snli/dev_p_h_syn_noun.csv"

    if transformation_type == "p_h":
        def train_trans(df): return path_base_transformation(
            df, train_path_mod)

        def dev_trans(df): return path_base_transformation(df, dev_path_mod)

    elif transformation_type == "only_p":
        def train_trans(df): return path_base_transformation_p(
            df, train_path_mod)

        def dev_trans(df): return path_base_transformation_p(df, dev_path_mod)

    elif transformation_type == "only_h":
        def train_trans(df): return path_base_transformation_h(
            df, train_path_mod)

        def dev_trans(df): return path_base_transformation_h(df, dev_path_mod)

    else:
        exit()

    dev_t = dev_trans(dev)

    # hyperparms for the LR models

    param_grid = {"C": np.linspace(0, 3, C_size),
                  "penalty": ["l2"]}

    hyperparams = {"RepresentationFunction": Tfidf,
                   "cv": cv,
                   "solver": 'lbfgs',
                   "random_state": None,
                   "verbose": False,
                   "n_jobs": n_jobs,
                   "n_iter": n_iter,
                   "max_features": max_features,
                   "label_translation": get_ternary_label,
                   "param_grid": param_grid}

    # performing the tests

    results = LIMts_test(train=train,
                         dev=dev,
                         train_transformation=train_trans,
                         dev_transformation=dev_trans,
                         rho=rho,
                         Model=LRWrapper,
                         hyperparams=hyperparams,
                         M=M,
                         E=E,
                         S=S,
                         verbose=verbose,
                         random_state=random_state)

    # saving results

    results.to_csv(results_path, index=False)


if __name__ == '__main__':

    debug = True

    pcts = [0.0, 0.25, 0.5, 0.75, 1.0]
    M = 10

    if debug:
        pcts = [0.3]
        M = 1

    for rho in pcts:

        print("\nPerforming tests for rho = {:.1%}\n".format(rho))

        wordsyn_test(transformation_type="p_h",
                     max_features=None,
                     C_size=500,
                     cv=5,
                     n_jobs=1,
                     n_iter=2,
                     rho=rho,
                     M=M,
                     E=5,
                     S=1000,
                     verbose=True,
                     random_state=1234,
                     debug=debug)
