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
from tqdm import tqdm


# variables
folder = "snli"
n_cores = 8
rho = 0
random_state = None
dgp_seed = None
n_inter = 10

output_dir_name = "bert_base_snli_search"

basic_hyperparams = {"local_rank": -1,
             "max_seq_length": 10,
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
             "transformation_name": None,
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


train = pd.read_csv("data/{}/train.csv".format(folder))
train = train.sample(50000)
train, dev_o = train_test_split(train, test_size=0.2)

print("clean train")
train = clean_df(train, n_cores=n_cores)

print("clean dev")
dev_o = clean_df(dev_o, n_cores=n_cores)


print("train.shape", train.shape)
print("dev.shape", dev_o.shape)


param_grid = {"max_seq_length": range(50, 210,10),
             "num_train_epochs": np.linspace(1,2.5, 10),       
             "learning_rate": np.linspace(0.00005,1, 10),
             "weight_decay": np.linspace(0,0.1, 10),
             "adam_epsilon": np.linspace(1e-8, 0.1, 10),
             "max_grad_norm": np.linspace(0.00005,1, 10)}


all_hypers = []
all_times = []
all_accs = []

for i in tqdm(range(n_inter)):
    hyperparams = basic_hyperparams.copy()
    for param in param_grid:
        hyperparams[param] =  np.random.choice(param_grid[param])
    if hyperparams["random_state"] is None:
        hyperparams["random_state"] = np.random.choice(range(1, 2333233))
    model = BertWrapper(hyperparams)
    init = time()
    model.fit(train)
    train_time = time() - init
    result = model.get_results(dev_o, mode="dev")
    acc = result.indicator.mean()
    all_hypers.append(hyperparams)
    all_times.append(train_time)
    all_accs.append(acc)
    del hyperparams, model

# Save best results
i = np.argmax(all_accs)
best_assigment = all_hypers[i]
mean_time = np.mean(all_times)
best_score = all_accs[i]


with open(output_dir_name + "/params.txt", "w") as file:
    for key in best_assigment:
        file.write("{} = {}\n".format(key, best_assigment[key]))
    file.write("\nbest_acc = {:.1%}".format(best_score))
    file.write("\nmean time = {:.1f} s".format(mean_time))
    file.write("\nnumber of search trials = {}".format(n_inter))

