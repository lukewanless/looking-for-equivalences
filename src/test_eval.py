#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertForSequenceClassification
from lr.models.transformers.processor import clean_df
from lr.text_processing.transformations.wordnet import path_base_transformation
from lr.stats.h_testing import h_test_transformer_trained_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import shutil
import os
from lr.models.transformers.BertWrapper import BertWrapper


# Variables

folder = "snli"
result_folder = "results/snli/bert/sin_p_h/"
transformation_name = "wordnet sin tranformation p and h"
n_cores = 16
rho = 0.75
dgp_seed = 224
random_state = 29
name = "rho_{:.2f}_dgp_seed_{}_random_state_{}".format(rho, dgp_seed, random_state)
name = name.replace(".", "p")
output_dir_name = "bert_p_h_" + name
save_path = output_dir_name + '/checkpoint-1500/'


# Perform Testing

dev_o = pd.read_csv("data/{}/dev.csv".format(folder))

print("clean dev")
dev_o = clean_df(dev_o, n_cores=n_cores)

# Transformations

train_path_mod = "data/{}/train_p_h_syn_noun.csv".format(folder)
dev_path_mod = "data/{}/dev_p_h_syn_noun.csv".format(folder)

def train_trans(df): return path_base_transformation(df, train_path_mod)
def dev_trans(df): return path_base_transformation(df, dev_path_mod)

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
               "max_steps": 1500,
               "warmup_steps": 0,
               "save_steps": 250,
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
               "base_path": "data/{}/cached_".format(folder)}


saved_model = BertWrapper(hyperparams)
saved_model.load(save_path)

print("testing")

test_results = h_test_transformer_trained_model(df_dev=dev_o,
                                                df_dev_t=dev_t,
                                                transformer=saved_model,
                                                hyperparams=hyperparams)


# show

def show_df(df):
    not_boot_columns = [c for c in df.columns if c.find("boot") == -1]
    print(df[not_boot_columns])
    

test_results.loc[0, "training_time"] = 4.032584
test_results.loc[0, "test_time"] = 4.549876


show_df(test_results)

# save

result_path  = "results/snli/bert/sin_p_h/new/" + name + ".csv"
test_results.to_csv(result_path, index=False)

print()
print("saved in {}".format(result_path))



