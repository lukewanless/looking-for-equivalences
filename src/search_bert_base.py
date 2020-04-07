import os
import numpy as np
import pandas as pd
from time import time
from lr.models.transformers.BertWrapper import BertWrapper
from lr.models.transformers.processor import clean_df
from lr.training.util import filter_df_by_label
from tqdm import tqdm
import glob
import argparse


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
           verbose):

    # Get data

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

    # Get hyperarams

    basic_hyperparams = {"local_rank": -1,
                         "overwrite_cache": False,
                         "per_gpu_train_batch_size": 32,
                         "per_gpu_eval_batch_size": 50,
                         "gradient_accumulation_steps": 1,
                          "max_steps": -1,
                         "warmup_steps": 0,
                         "save_steps": 80580,
                         "no_cuda": False,
                         "n_gpu": 1,
                         "model_name_or_path": "bert",
                         "output_dir": output_dir_name,
                         "random_state": random_state,
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

    param_grid = {"max_seq_length": range(50, 210, 10),
                  "num_train_epochs": np.linspace(1, 2.5, 10),
                  "learning_rate": np.linspace(0.00005, 1, 10),
                  "weight_decay": np.linspace(0, 0.1, 10),
                  "adam_epsilon": np.linspace(1e-8, 0.1, 10),
                  "max_grad_norm": np.linspace(0.00005, 1, 10)}

    np.random.seed(random_state)
    choices = []

    for i in range(n_iter):
        hyper_choice = {}
        for k in param_grid:
            hyper_choice[k] = np.random.choice(param_grid[k])
        choices.append(hyper_choice)

    # Search

    all_accs = []
    all_train_times = []
    init_search = time()

    for hyper_choice in tqdm(choices):
        hyperparams = basic_hyperparams.copy()
        hyperparams.update(hyper_choice)
        model = BertWrapper(hyperparams)
        init = time()
        model.fit(train)
        train_time = time() - init
        result = model.get_results(dev, mode="dev")
        acc = result.indicator.mean()
        all_accs.append(acc)
        all_train_times.append(train_time)
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
    output_dir_name = "hyperparams/bert_base_snli"

    search(train_path=train_path,
           dev_path=dev_path,
           random_state=args.random_state,
           folder=folder,
           n_iter=args.n_iter,
           n_cores=args.n_cores,
           output_dir_name=output_dir_name,
           verbose=True)
