import os
import numpy as np
import pandas as pd
from time import time
from lr.models.transformers.RobertaWrapper import RobertaWrapper
from lr.models.transformers.processor import clean_df
from lr.training.util import filter_df_by_label
from tqdm import tqdm
import glob
import argparse
import logging


# Help Functions


def clean_folder(folder):
    cacheds = glob.glob('data/{}/cached_*'.format(folder))
    for path in cacheds:
        os.remove(path)


def clean_folder_log(output_dir_name):
    cacheds = glob.glob(output_dir_name + '/*_log.csv')
    for path in cacheds:
        os.remove(path)


def search(train_path,
           dev_path,
           folder,
           random_state,
           n_cores,
           n_iter,
           output_dir_name,
           verbose,
           max_range=10):

    # Get data

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    # dev = dev.sample(1000)  # debug

    if verbose:
        print("clean train")
    train = clean_df(train, n_cores=n_cores)

    if verbose:
        print("clean dev")
    dev = clean_df(dev, n_cores=n_cores)

    if verbose:
        print("train.shape", train.shape)
        print("dev.shape", dev.shape)

    # Get hyperarams

    basic_hyperparams = {"local_rank": -1,
                         "overwrite_cache": False,
                         "per_gpu_train_batch_size": 32,
                         "per_gpu_eval_batch_size": 50,
                         "gradient_accumulation_steps": 1,
                         "max_steps": -1,
                         # "max_steps": 100,  # debug
                         "warmup_steps": 0,
                         "save_steps": 80580,
                         "no_cuda": False,
                         "n_gpu": 1,
                         "model_name_or_path": "roberta",
                         "output_dir": output_dir_name,
                         "random_state": random_state,
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

    choice_0 = {'num_train_epochs': 1.0,
                "max_seq_length": 200,
                "learning_rate": 5e-5,
                "weight_decay": 0.0,
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0}

    param_grid = {"max_seq_length": range(50, 210, max_range),
                  "num_train_epochs": [1,2,3],
                  "learning_rate": np.linspace(0.00005, 0.0001, max_range),
                  "weight_decay": np.linspace(0, 0.01, max_range),
                  "adam_epsilon": np.linspace(1e-8, 1e-7, max_range),
                  "max_grad_norm": np.linspace(0.9, 1.0, max_range)}

    np.random.seed(random_state)
    choices = []

    for i in range(n_iter):
        hyper_choice = {}
        for k in param_grid:
            hyper_choice[k] = np.random.choice(param_grid[k])
        choices.append(hyper_choice)

#     choices.append(choice_0)

    # Search

    all_accs = []
    all_train_times = []
    init_search = time()

    for hyper_choice in tqdm(choices):
        hyperparams = basic_hyperparams.copy()
        hyperparams.update(hyper_choice)
        model = RobertaWrapper(hyperparams)
        init = time()
        model.fit(train)
        train_time = time() - init
        result = model.get_results(dev, mode="dev")
        acc = result.indicator.mean()
        all_accs.append(acc)
        all_train_times.append(train_time)

        # log partial Results
        logging.info("\n\n\n***** acc = {:.1%} *****\n".format(acc))
        logging.info(
            "***** train_time = {:.2f} *****\n".format(train_time / 3600))
        for k in hyper_choice:
            logging.info("***** {} = {} *****\n".format(k, hyper_choice[k]))
        logging.info("\n\n\n")

        clean_folder(folder)

    search_time = time() - init_search
    search_time = search_time / 3600

    # Store Results

    best_id = np.argmax(all_accs)
    best_score = all_accs[best_id]
    param_df = pd.DataFrame(choices[best_id], index=[0])

    dict_ = {"search_random_state": [random_state],
             "number_of_search_trails": [n_iter],
             "expected_val_score": [np.mean(all_accs)],
             "best_val_score": [best_score],
             "mean_fit_time": [np.mean(all_train_times) / 3600],
             "search_time": [search_time]}
    search_results = pd.DataFrame(dict_)
    search_results = pd.merge(
        search_results,
        param_df,
        left_index=True,
        right_index=True)
    path = output_dir_name + "/search_{}.csv".format(random_state)
    clean_folder_log(output_dir_name)
    search_results.to_csv(path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('random_state',
                        type=int,
                        help='random_state')

    parser.add_argument('n_iter',
                        type=int,
                        help='number of iterations')

    parser.add_argument('n_cores',
                        type=int,
                        help='number of cores')
    args = parser.parse_args()

    folder = "snli"
    train_path = "data/{}/train_sample.csv".format(folder)
    dev_path = "data/{}/dev.csv".format(folder)
    output_dir_name = "hyperparams/roberta_base_snli"

    search(train_path=train_path,
           dev_path=dev_path,
           random_state=args.random_state,
           folder=folder,
           n_iter=args.n_iter,
           n_cores=args.n_cores,
           output_dir_name=output_dir_name,
           verbose=True)

