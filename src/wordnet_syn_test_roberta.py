from lr.models.transformers.processor import clean_df
from lr.models.transformers.train_functions import set_seed
from lr.models.transformers.RobertaWrapper import RobertaWrapper
from lr.text_processing.transformations.wordnet import path_base_transformation
from lr.text_processing.transformations.wordnet import path_base_transformation_p
from lr.text_processing.transformations.wordnet import path_base_transformation_h
from lr.stats.h_testing import DGP, h_test_transformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import shutil
import os
import argparse


def run_test(rho, dgp_seed, random_state, n_cores):

    # Variables
    folder = "snli"
    result_folder = "results/snli/roberta_base/sin_p_h/"

    transformation_name = "wordnet sin tranformation p and h"
    # transformation_name = "wordnet sin tranformation p"
    # transformation_name = "wordnet sin tranformation h"

    name = "rho_{:.2f}_dgp_seed_{}_random_state_{}".format(
        rho, dgp_seed, random_state)
    name = name.replace(".", "p")
    output_dir_name = "roberta_base_p_h_" + name
    # output_dir_name = "bert_h_" + name

    # Data

    train = pd.read_csv("data/{}/train.csv".format(folder))
    dev_o = pd.read_csv("data/{}/dev.csv".format(folder))

    print("clean train")
    train = clean_df(train, n_cores=n_cores)

    print("clean dev")
    dev_o = clean_df(dev_o, n_cores=n_cores)

    # Transformations

    train_path_mod = "data/{}/train_p_h_syn_noun.csv".format(folder)
    dev_path_mod = "data/{}/dev_p_h_syn_noun.csv".format(folder)

    def train_trans(df): return path_base_transformation(df, train_path_mod)
    def dev_trans(df): return path_base_transformation(df, dev_path_mod)

    # def train_trans(df): return path_base_transformation_p(df, train_path_mod)
    # def dev_trans(df): return path_base_transformation_p(df, dev_path_mod)

    # def train_trans(df): return path_base_transformation_h(df, train_path_mod)
    # def dev_trans(df): return path_base_transformation_h(df, dev_path_mod)

    print("transform dev")
    dev_t = dev_trans(dev_o)

    # Hyperparams

    hyperparams = {"local_rank": -1,
                   "max_seq_length": 200,
                   "overwrite_cache": False,
                   "num_train_epochs": 1.0,
                   "per_gpu_train_batch_size": 32,
                   "per_gpu_eval_batch_size": 50,
                   "gradient_accumulation_steps": 1,
                   "learning_rate": 5e-5,
                   "weight_decay": 0.0,
                   "adam_epsilon": 1e-8,
                   "max_grad_norm": 1.0,
                   "max_steps": 1505,
                   "warmup_steps": 0,
                   "save_steps": 1500,
                   "no_cuda": False,
                   "n_gpu": 1,
                   "data_set_name": folder,
                   "transformation_name": transformation_name,
                   "number_of_simulations": 1000,
                   "rho": rho,
                   "model_name_or_path": "roberta",
                   "output_dir": output_dir_name,
                   "random_state": random_state,
                   "dgp_seed": dgp_seed,
                   "fp16": False,
                   "fp16_opt_level": "01",
                   "device": "cpu",
                   "verbose": True,
                   "model_type": "roberta",
                   "pad_on_left": False,
                   "pad_token": 0,
                   "n_cores": n_cores,
                   'eval_sample_size': 200,
                   "pad_token_segment_id": 0,
                   "mask_padding_with_zero": True,
                   "base_path": "data/{}/cached_".format(folder),
                   "pretrained_weights": 'roberta-base'}

    # Selecting one data by DGP

    dgp_seed = hyperparams["dgp_seed"]
    rho = hyperparams["rho"]
    rs = hyperparams["random_state"]

    set_seed(dgp_seed, 0)
    dgp = DGP(train, train_trans, rho=rho)
    train_ = dgp.sample_transform()

    # Testing

    print("testing")

    test_results = h_test_transformer(df_train=train_,
                                      df_dev=dev_o,
                                      df_dev_t=dev_t,
                                      ModelWrapper=RobertaWrapper,
                                      hyperparams=hyperparams)

    # Saving Results

    result_path = result_folder + name
    result_path = result_path.replace(".", "p") + ".csv"
    test_results.to_csv(result_path, index=False)

    print()
    print("results in {}".format(result_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument('rho',
                        type=float,
                        help='rho')

    parser.add_argument('dgp_seed',
                        type=int,
                        help='dgp_seed for training sample')

    parser.add_argument('random_state',
                        type=int,
                        help='random_state for model training')

    parser.add_argument('n_cores',
                        type=int,
                        help='number of cores')
    args = parser.parse_args()
    
    run_test(rho=args.rho,
             dgp_seed=args.dgp_seed,
             random_state=args.random_state,
             n_cores=args.n_cores)
