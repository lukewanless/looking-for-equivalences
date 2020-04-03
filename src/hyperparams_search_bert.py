from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from lr.models.transformers.processor import clean_df
from lr.models.transformers.BertWrapper import BertWrapper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import shutil
import os
import argparse


def run_search(rho, n_iter, n_cores):

    # Variables
    # folder = "snli"
    folder = "toy"

    # result_folder = "results/snli/roberta_base/sin_p/"
    result_folder = "results/snli/bert_base/sin_p_h/"

    output_dir_name = "bert_base_snli_search"

    # Data

    train = pd.read_csv("data/{}/train.csv".format(folder))
    train, dev_o = train_test_split(train, test_size=0.2)

    print("clean train")
    train = clean_df(train, n_cores=n_cores)

    print("clean dev")
    dev_o = clean_df(dev_o, n_cores=n_cores)


    for _ in range(n_iter):


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
                     "max_steps": -1,
                     "warmup_steps": 0,
                     "save_steps": 8580,
                     "no_cuda": False,
                     "n_gpu": 1,
                     "data_set_name": folder,
                     "transformation_name": transformation_name,
                     "number_of_simulations": 1000,
                     "rho": rho,
                     "model_name_or_path": "bert",
                     "output_dir": output_dir_name,
                     "random_state": random_state,
                     "dgp_seed": dgp_seed,
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
                     "pretrained_weights": 'bert-base-uncased'}

    # Selecting one data by DGP

            self.model = RandomizedSearchCV(gbm,
                                            param_distributions=param_grid,
                                            cv=cv,
                                            n_iter=n_iter,
                                            random_state=random_state,
                                            verbose=verbose,
                                            n_jobs=n_jobs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

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
