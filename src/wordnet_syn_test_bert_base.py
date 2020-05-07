import argparse
import os
import numpy as np
import pandas as pd
from time import time
import glob
from lr.models.transformers.BertWrapper import BertWrapper
from lr.text_processing.util import pre_process_nli_df
from lr.training.util import get_ternary_label, filter_df_by_label
from lr.text_processing.transformations.wordnet import path_base_transformation
from lr.stats.h_testing import DGP
from lr.stats.h_testing import get_matched_results_transformers
from lr.stats.h_testing import get_paired_t_statistic
from lr.stats.h_testing import get_cochran_statistic
from lr.stats.h_testing import get_boots_series_under_H0
from lr.stats.h_testing import get_boot_paired_t_p_value
from lr.stats.h_testing import get_boot_cochran_p_value


def clean_folder_log(output_dir_name):
    cacheds = glob.glob(output_dir_name + '/*_log.csv')
    for path in cacheds:
        os.remove(path)


def clean_folder(folder):
    cacheds = glob.glob('data/{}/cached_*'.format(folder))
    for path in cacheds:
        os.remove(path)


def run_test(folder,
             train_path,
             dev_plus_path,
             test_path,
             transformation_name,
             train_path_mod,
             dev_plus_mod,
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
             output_dir,
             n_cores,
             verbose,
             save_steps=8500,
             clean=True):

    # Get data

    init_test = time()

    train = pd.read_csv(train_path)
    dev_plus = pd.read_csv(dev_plus_path)
    test = pd.read_csv(test_path)

    train = filter_df_by_label(train.dropna()).reset_index(drop=True)
    dev_plus = filter_df_by_label(dev_plus.dropna()).reset_index(drop=True)
    test = filter_df_by_label(test.dropna()).reset_index(drop=True)

    pre_process_nli_df(train)
    pre_process_nli_df(dev_plus)
    pre_process_nli_df(test)

    # Get hyperparams

    params_keys = ['num_train_epochs', "max_seq_length",
                   "learning_rate", "weight_decay",
                   "adam_epsilon", "max_grad_norm"]

    hyperparams = {"local_rank": -1,
                   "overwrite_cache": False,
                   "per_gpu_train_batch_size": 32,
                   "per_gpu_eval_batch_size": 50,
                   "gradient_accumulation_steps": 1,
                   # "max_steps": 50,  # debug
                   "max_steps": -1,
                   "warmup_steps": 0,
                   "save_steps": save_steps,
                   "no_cuda": False,
                   "n_gpu": 1,
                   "data_set_name": folder,
                   "transformation_name": transformation_name,
                   "rho": rho,
                   "model_name_or_path": "bert",
                   "output_dir": output_dir,
                   "fp16": False,
                   "fp16_opt_level": "01",
                   "device": "cpu",
                   "verbose": True,
                   "model_type": "bert",
                   "pad_on_left": False,
                   "pad_token": 0,
                   "n_cores": n_cores,
                   'eval_sample_size': 200,
                   "pad_token_segment_id": 0,
                   "mask_padding_with_zero": True,
                   "base_path": "data/{}/cached_".format(folder),
                   "pretrained_weights": 'bert-base-uncased',
                   "number_of_simulations": number_of_simulations,
                   "search_random_state": search_random_state,
                   "dgp_random_state": dgp_random_state,
                   "train_random_state": train_random_state,
                   "random_state": train_random_state,
                   "boot_random_state": boot_random_state,
                   "output_raw_result": output_raw_result,
                   "output_result": output_result}

    search_results = pd.read_csv(search_path)

    for k in params_keys:
        hyperparams[k] = search_results.loc[0, k]

    # Set transformed version of the datasets

    def train_trans(df): return path_base_transformation(df, train_path_mod)

    def dev_trans(df): return path_base_transformation(df, dev_plus_mod)

    def test_trans(df): return path_base_transformation(df, test_path_mod)

    test_t = test_trans(test)

    dev_plus_t = dev_trans(dev_plus)

    # get_training_sample

    train.loc[:, "o_index"] = train.index.values

    dgp_train = DGP(data=train,
                    transformation=train_trans,
                    rho=rho)

    train_ = dgp_train.sample_transform(random_state=dgp_random_state)

    # Train

    model = BertWrapper(hyperparams)
    # _, _, train_time = model.fit(train_.sample(1000, random_state=10))  #
    # debug
    _, _, train_time = model.fit(train_)

    # # Test set Eval
    # test_results = model.get_results(test.iloc[:1000], mode="test")  # debug
    # test_t_results = model.get_results(
    #     test_t.iloc[:1000], mode="test_t")  # debug

    test_results = model.get_results(test, mode="test")
    test_t_results = model.get_results(test_t, mode="test_t")

    # # Dev set Eval
    # dev_results = model.get_results(
    #     dev_plus.iloc[:1000], mode="dev_plus")  # debug
    # dev_t_results = model.get_results(
    #     dev_plus_t.iloc[:1000], mode="dev_plus_t")  # debug

    dev_results = model.get_results(dev_plus, mode="dev_plus")
    dev_t_results = model.get_results(dev_plus_t, mode="dev_plus_t")

    # Getting statistics

    m_results = get_matched_results_transformers(test_results, test_t_results)

    test_acc = m_results.A.mean()
    transformed_test_acc = m_results.B.mean()
    t_obs, acc_diff, test_size, standart_error = get_paired_t_statistic(
        m_results)
    cochran_obs = get_cochran_statistic(m_results)

    dev_m_results = get_matched_results_transformers(
        dev_results, dev_t_results)
    dev_acc = dev_m_results.A.mean()
    dev_t_acc = dev_m_results.B.mean()
    dev_diff = np.abs(dev_acc - dev_t_acc)

    # get simulations

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
             "dev_plus_accuracy": [dev_acc],
             "transformed_dev_plus_accuracy": [dev_t_acc],
             "dev_plus_accuracy_difference": [dev_diff],
             "training_time": [train_time / 3600],
             "test_time": [htest_time / 3600]}

    test_results = pd.DataFrame(dict_)
    m_results.to_csv(output_raw_result, index=False)
    test_results.to_csv(output_result, index=False)
    if verbose:
        print(output_raw_result)
        print(output_result)
    if clean:
        clean_folder_log(output_dir)
        clean_folder(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('folder',
                        type=str,
                        help='data folder')

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

    folder = args.folder
    rho = args.rho
    search_random_state = args.search_random_state
    dgp_random_state = args.dgp_random_state
    train_random_state = args.train_random_state
    boot_random_state = args.boot_random_state
    n_cores = args.n_cores

    number_of_simulations = 1000
    verbose = True

    train_path = "data/{}/train_sample.csv".format(folder)
    dev_plus_path = "data/{}/train_not_in_sample.csv".format(folder)
    test_path = "data/{}/test.csv".format(folder)

    train_path_mod = "data/{}/train_sample_p_h_syn_noun.csv".format(folder)
    dev_plus_mod = "data/{}/train_not_in_sample_p_h_syn_noun.csv".format(
        folder)
    test_path_mod = "data/{}/test_p_h_syn_noun.csv".format(folder)

    search_path = "hyperparams/bert_base_{}/search_{}.csv".format(folder,
                                                                  search_random_state)
    assert os.path.exists(search_path)

    transformation_name = "wordnet syn tranformation p and h"
    output_raw_result = "raw_results/{}/bert_base/syn_p_h/rho_{:.2f}_results".format(folder,
                                                                                     rho)
    output_raw_result = output_raw_result.replace(".", "p") + ".csv"
    output_result = "results/{}/bert_base/syn_p_h/rho_{:.2f}_results".format(folder,
                                                                             rho)
    output_result = output_result.replace(".", "p") + ".csv"
    output_dir = "results/{}/bert_base/syn_p_h/".format(folder)

    run_test(folder=folder,
             train_path=train_path,
             dev_plus_path=dev_plus_path,
             test_path=test_path,
             transformation_name=transformation_name,
             train_path_mod=train_path_mod,
             dev_plus_mod=dev_plus_mod,
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
             output_dir=output_dir,
             n_cores=n_cores,
             verbose=verbose)
