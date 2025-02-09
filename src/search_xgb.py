from lr.models.transformers.processor import clean_df
from lr.training.language_representation import Tfidf
from lr.training.util import get_ternary_label, filter_df_by_label
from lr.models.xgb import XGBCWrapper
import numpy as np
import pandas as pd
from time import time
import shutil
import os
import argparse


# Variables and basic params

def search(train_path,
           dev_path,
           random_state,
           n_cores, cv,
           n_iter, output_dir_name,
           verbose):

    # hyperparms for the XGB models
    param_grid = {'n_estimators': range(10, 31, 1),
                  'max_depth': range(2, 21),
                  "reg_alpha": np.arange(0.05, 1.05, 0.05),
                  "reg_gamma": np.arange(0.05, 1.05, 0.05),
                  "learning_rate": np.arange(0.05, 1.05, 0.05),
                  "subsample": np.arange(0.05, 1.05, 0.05),
                  "colsample_bytree": np.arange(0.05, 1.05, 0.05)}

    hyperparams = {"RepresentationFunction": Tfidf,
                   "cv": cv,
                   "random_state": random_state,
                   "n_jobs": n_cores,
                   "n_iter": n_iter,
                   "max_features": None,
                   "label_translation": get_ternary_label,
                   "param_grid": param_grid,
                   "data_set_name": "snli",
                   "verbose": verbose}

    # Set Data

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)

    if verbose:
        print("clean train")
    train = clean_df(train, n_cores=n_cores)

    if verbose:
        print("clean dev")
    dev = clean_df(dev, n_cores=n_cores)

    if verbose:
        print("train.shape", train.shape)
        print("dev.shape", dev.shape)

    # Search

    model = XGBCWrapper(hyperparams)
    init = time()
    model.fit(train)
    search_time = time() - init
    search_time = search_time / 3600
    best_score = model.get_acc(dev)
    raw_results = pd.DataFrame(model.model.cv_results_)

    # Store Results

    dict_ = {"search_random_state": [random_state],
             "number_of_search_trails": [n_iter],
             "expected_val_score": [raw_results.mean_test_score.mean()],
             "best_val_score": [best_score],
             "mean_fit_time": [raw_results.mean_fit_time.mean() / 3600],
             "search_time": [search_time]}
    search_results = pd.DataFrame(dict_)
    param_df = pd.DataFrame(model.model.best_params_, index=[0])
    search_results = pd.merge(
        search_results,
        param_df,
        left_index=True,
        right_index=True)

    path = output_dir_name + "/search_{}.csv".format(random_state)

    search_results.to_csv(path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('folder',
                        type=str,
                        help='data folder')

    parser.add_argument('random_state',
                        type=int,
                        help='random_state')

    parser.add_argument('cv',
                        type=int,
                        help='cross validation')

    parser.add_argument('n_iter',
                        type=int,
                        help='number of iterations')

    parser.add_argument('n_cores',
                        type=int,
                        help='number of cores')
    args = parser.parse_args()

    folder = args.folder
    train_path = "data/{}/train_sample.csv".format(folder)
    dev_path = "data/{}/dev.csv".format(folder)
    output_dir_name = "hyperparams/xgb_{}".format(folder)

    search(train_path=train_path,
           dev_path=dev_path,
           random_state=args.random_state,
           cv=args.cv,
           n_iter=args.n_iter,
           n_cores=args.n_cores,
           output_dir_name=output_dir_name,
           verbose=True)
