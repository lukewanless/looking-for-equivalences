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


def run_test(train_path,
             train_path_mod,
             output_path,
             rho,
             dgp_random_state):

    # Get data

    init_test = time()

    train = pd.read_csv(train_path)

    train = filter_df_by_label(train.dropna()).reset_index(drop=True)

    pre_process_nli_df(train)

    # Set transformed version of the datasets

    def train_trans(df): return path_base_transformation(df, train_path_mod)

    # get_training_sample

    train.loc[:, "o_index"] = train.index.values

    dgp_train = DGP(data=train,
                    transformation=train_trans,
                    rho=rho)

    train_ = dgp_train.sample_transform(random_state=dgp_random_state)

    train_.to_csv(output_path, index=False)

    htest_time = time() - init_test
    print(htest_time)

    # if verbose:
        # print(output_raw_result)
        # print(output_result)
    # if clean:
    #     clean_folder_log(output_dir)
    #     clean_folder(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('rho',
                        type=float,
                        help='modification percentage for train')

    parser.add_argument('experiment_number',
                        type=int,
                        help='experiment number')

    parser.add_argument('dgp_random_state',
                        type=int,
                        help='random_state for dgp sampling')

    args = parser.parse_args()

    # folder = args.folder
    rho = args.rho
    experiment_number = args.experiment_number
    dgp_random_state = args.dgp_random_state
    # n_cores = args.n_cores

    rho_name = str(rho).split('.')[-1]
    if rho_name != '0':
        rho_name += '0'

    if rho == 1.0:
        rho_name = '100'

    number_of_simulations = 1000
    verbose = True

    train_path = "data/sick/train.csv"

    train_path_mod = "data/sick/train_p_h_syn_noun.csv"

    output_path = "data/sick/rho{}/train_rho{}-{}.csv".format(rho_name,rho_name,experiment_number)

    run_test(train_path=train_path,
             train_path_mod=train_path_mod,
             output_path=output_path,
             rho=rho,
             dgp_random_state=dgp_random_state)
