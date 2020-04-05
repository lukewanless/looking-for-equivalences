import os
import numpy as np
import pandas as pd
from time import time
import argparse

from lr.stats.h_testing import DGP
from lr.stats.h_testing import get_matched_results
from lr.stats.h_testing import get_paired_t_statistic
from lr.stats.h_testing import get_cochran_statistic
from lr.stats.h_testing import get_boots_series_under_H0
from lr.stats.h_testing import get_boot_paired_t_p_value
from lr.stats.h_testing import get_boot_cochran_p_value

from lr.models.xgb import XGBCWrapper
from lr.training.language_representation import Tfidf
from lr.text_processing.util import pre_process_nli_df
from lr.training.util import get_ternary_label, filter_df_by_label
from lr.text_processing.transformations.wordnet import path_base_transformation


def run_test(train_path,
             dev_path,
             test_path,
             transformation_name,
             train_path_mod,
             dev_path_mod,
             test_path_mod,
             search_path,
             rho,
             search_random_state,
             train_random_state,
             boot_random_state,
             dgp_random_state,
             number_of_simulations,
             output_raw_result,
             output_result,
             n_cores,
             verbose):

    # Get data

    init_test = time()

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)

    train = filter_df_by_label(train.dropna()).reset_index(drop=True)
    dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
    test = filter_df_by_label(test.dropna()).reset_index(drop=True)

    pre_process_nli_df(train)
    pre_process_nli_df(dev)

    # Get hyperarams

    params_keys = ['n_estimators', 'max_depth', "reg_alpha",
                   "reg_gamma", "learning_rate", "subsample", "colsample_bytree"]

    hyperparams = {"RepresentationFunction": Tfidf,
                   "n_jobs": n_cores,
                   "max_features": None,
                   "label_translation": get_ternary_label,
                   "data_set_name": "snli",
                   "transformation_name": transformation_name,
                   "rho": rho,
                   "model_name_or_path": "gradient boosting",
                   "number_of_simulations": number_of_simulations,
                   "search_random_state": search_random_state,
                   "dgp_random_state": dgp_random_state,
                   "train_random_state": train_random_state,
                   "boot_random_state": boot_random_state,
                   "output_raw_result": output_raw_result,
                   "output_result": output_result,
                   "verbose": verbose}

    search_results = pd.read_csv(search_path)

    for k in params_keys:
        hyperparams[k] = search_results.loc[0, k]

    # Set transformed version of the datasets

    def train_trans(df): return path_base_transformation(df, train_path_mod)

    def dev_trans(df): return path_base_transformation(df, dev_path_mod)

    def test_trans(df): return path_base_transformation(df, test_path_mod)

    dgp_train = DGP(data=train,
                    transformation=train_trans,
                    rho=rho)

    dgp_dev = DGP(data=dev,
                  transformation=dev_trans,
                  rho=rho)

    dgp_random_state = hyperparams["dgp_random_state"]

    train_t = dgp_train.sample(random_state=dgp_random_state)
    dev_t = dgp_dev.sample(random_state=dgp_random_state)
    test_t = test_trans(test)

    # we will train the xgb with train and dev data
    full_train_t = pd.concat([train_t, dev_t]).reset_index(drop=True)

    # Training the model

    model = XGBCWrapper(hyperparams)
    init_train = time()
    model.fit(full_train_t)
    train_time = time() - init_train

    # Get matched eval and relevant statistics

    m_results = get_matched_results(test,
                                    test_t,
                                    model,
                                    hyperparams["label_translation"])

    test_acc = m_results.A.mean()
    transformed_test_acc = m_results.B.mean()
    t_obs, acc_diff, test_size, standart_error = get_paired_t_statistic(
        m_results)
    cochran_obs = get_cochran_statistic(m_results)

    # Get p-values by bootstrap replications

    number_of_simulations = hyperparams["number_of_simulations"]
    boot_random_state = hyperparams["boot_random_state"]

    def get_paired_t(matched_results):
        t_obs, _, _, _ = get_paired_t_statistic(matched_results)
        return t_obs

    paired_t_boots = get_boots_series_under_H0(m_results,
                                               get_paired_t,
                                               number_of_simulations,
                                               boot_random_state)

    cochran_boots = get_boots_series_under_H0(m_results,
                                              get_cochran_statistic,
                                              number_of_simulations,
                                              boot_random_state)

    paired_t_p_value = get_boot_paired_t_p_value(paired_t_boots, t_obs)

    cochran_p_value = get_boot_cochran_p_value(cochran_boots, cochran_obs)

    htest_time = time() - init_test

    # Aggregate all results

    dict_ = {"data": [hyperparams["data_set_name"]],
             "model": [hyperparams["model_name_or_path"]],
             "transformation": [hyperparams["transformation_name"]],
             "rho": [rho],
             "search_random_state": [hyperparams["search_random_state"]],
             "dgp_random_state": [dgp_random_state],
             "train_random_state": [hyperparams["train_random_state"]],
             "boot_random_state": [boot_random_state],
             "number_of_simulations": [number_of_simulations],
             "test_accuracy": [test_acc],
             "transformed_test_accuracy": [transformed_test_acc],
             "accuracy_difference": [acc_diff],
             "test_size": [test_size],
             "standart_error": [standart_error],
             "observable_paired_t_stats": [t_obs],
             "paired_t_p_value": [paired_t_p_value],
             "observable_cochran_stats": [cochran_obs],
             "cochran_p_value": [cochran_p_value],
             "training_time": [train_time / 3600],
             "test_time": [htest_time / 3600]}

    test_results = pd.DataFrame(dict_)
    m_results.to_csv(output_raw_result, index=False)
    test_results.to_csv(output_result, index=False)
    if verbose:
        print(output_raw_result)
        print(output_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('rho',
                        type=float,
                        help='modification percentage for train')

    parser.add_argument('search_random_state',
                        type=int,
                        help='random_state for hyperparams search')

    parser.add_argument('dgp_random_state',
                        type=int,
                        help='random_state for dgp sampling')

    parser.add_argument('train_random_state',
                        type=int,
                        help='random_state for model fitting')

    parser.add_argument('boot_random_state',
                        type=int,
                        help='random_state for bootstrap simulation')

    parser.add_argument('n_cores',
                        type=int,
                        help='number of cores')
    args = parser.parse_args()

    rho = args.rho
    search_random_state = args.search_random_state
    dgp_random_state = args.dgp_random_state
    train_random_state = args.train_random_state
    boot_random_state = args.boot_random_state
    n_cores = args.n_cores

    number_of_simulations = 1000
    verbose = True

    train_path = "data/snli/train.csv"
    dev_path = "data/snli/dev.csv"
    test_path = "data/snli/test.csv"

    train_path_mod = "data/snli/train_p_h_syn_noun.csv"
    dev_path_mod = "data/snli/dev_p_h_syn_noun.csv"
    test_path_mod = "data/snli/test_p_h_syn_noun.csv"

    search_path = "hyperparams/xgb_snli/search_{}.csv".format(
        search_random_state)
    assert os.path.exists(search_path)

    transformation_name = "wordnet syn tranformation p and h"
    output_raw_result = "raw_results/snli/xgb/syn_p_h/rho_{}_results".format(
        rho)
    output_raw_result = output_raw_result.replace(".", "p") + ".csv"
    output_result = "results/snli/xgb/syn_p_h/rho_{}_results".format(rho)
    output_result = output_result.replace(".", "p") + ".csv"

    run_test(train_path=train_path,
             dev_path=dev_path,
             test_path=test_path,
             transformation_name=transformation_name,
             train_path_mod=train_path_mod,
             dev_path_mod=dev_path_mod,
             test_path_mod=test_path_mod,
             search_path=search_path,
             rho=rho,
             search_random_state=search_random_state,
             train_random_state=train_random_state,
             dgp_random_state=dgp_random_state,
             boot_random_state=boot_random_state,
             number_of_simulations=number_of_simulations,
             output_raw_result=output_raw_result,
             output_result=output_result,
             n_cores=n_cores,
             verbose=verbose)
