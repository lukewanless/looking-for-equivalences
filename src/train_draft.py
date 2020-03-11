from lr.models.transformers.processor import *
from lr.models.transformers.util import evaluate, train
import logging
import os
import shutil
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from time import time
from sklearn.model_selection import train_test_split


folder = "snli"

hyperparams = {"local_rank": -1,
               "max_seq_length": 200,
               "overwrite_cache": False,
               "num_train_epochs": 1.0,
               "per_gpu_train_batch_size": 32,
               "per_gpu_eval_batch_size": 32,
               "gradient_accumulation_steps": 1,
               "learning_rate": 5e-5,
               "weight_decay": 0.0,
               "adam_epsilon": 1e-8,
               "max_grad_norm": 1.0,
               "max_steps": 1200,
               "warmup_steps": 0,
               "save_steps": 100,
               "no_cuda": False,
               "n_gpu": 1,
               "model_name_or_path": "bert",
               "output_dir": "bert",
               "random_state": 42,
               "fp16": False,
               "fp16_opt_level": "01",
               "device": "cpu",
               "verbose": True,
               "model_type": "bert",
               "pad_on_left": False,
               "pad_token": 0,
               "pad_token_segment_id": 0,
               "mask_padding_with_zero": True,
               "base_path": "data/{}/cached_".format(folder)}


# # loading tokenizers

# In[3]:


set_seed(hyperparams["random_state"], hyperparams["n_gpu"])

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
hyperparams["tokenizer"] = tokenizer


# ## Set results dict

# In[4]:


meta_results = {"moment": [],
                "type": [],
                "loss": [],
                "acc": [],
                "time": []}


# # df

# In[5]:


train_path = "data/{}/train.csv".format(folder)
set_seed(hyperparams["random_state"], hyperparams["n_gpu"])

eval_sample_size = 200


df = pd.read_csv(train_path)


df_train, df_dev = train_test_split(df, test_size=0.2)
df_train_to_eval = df_train.sample(
    n=eval_sample_size,
    random_state=hyperparams["random_state"])
df_dev_to_eval = df_dev.sample(
    n=eval_sample_size,
    random_state=hyperparams["random_state"])


# ## Creating features


processor = NLIProcessor(hyperparams)
init = time()
train_cached_features_file = processor.df2features(df=df_train,
                                                   n_cores=8,
                                                   mode="train")

train_to_eval_cached_features_file = processor.df2features(df=df_train_to_eval,
                                                           n_cores=8,
                                                           mode="train_to_eval")

dev_cached_features_file = processor.df2features(df=df_dev,
                                                 n_cores=8,
                                                 mode="dev")

dev_to_eval_cached_features_file = processor.df2features(df=df_dev_to_eval,
                                                         n_cores=8,
                                                         mode="dev_to_eval")

p_time = time() - init
print("df 2 features | total time = {:.3f}".format(p_time / 60))


# ## Loading Datasets

init = time()
train_dataset = features2dataset(train_cached_features_file)
train_dataset_to_eval = features2dataset(train_to_eval_cached_features_file)
dev_dataset = features2dataset(dev_cached_features_file)
dev_dataset_to_eval = features2dataset(dev_to_eval_cached_features_file)
p_time = time() - init
print("features 2 Datasets | total time = {:.3f}".format(p_time / 60))


# ## Loading Model

model = BertForSequenceClassification.from_pretrained(
    pretrained_weights, num_labels=3)


# ### Eval before training
#
# #### Train sample

train_loss, train_results = evaluate(train_dataset_to_eval, hyperparams, model)
train_acc = (train_results.prediction == train_results.label).mean()

lmap = processor.get_label_map()
filtered = filter_df_by_label(df_train_to_eval.dropna()).reset_index(drop=True)
assert np.all(filtered.label.map(lambda x: lmap[x]) == train_results.label)


meta_results["moment"].append("before")
meta_results["type"].append("train")
meta_results["loss"].append(train_loss)
meta_results["acc"].append(train_acc)
meta_results["time"].append(np.nan)


# #### Dev sample


dev_loss, results = evaluate(dev_dataset_to_eval, hyperparams, model)
dev_acc = (results.prediction == results.label).mean()


filtered = filter_df_by_label(df_dev_to_eval.dropna()).reset_index(drop=True)
assert np.all(filtered.label.map(lambda x: lmap[x]) == results.label)


meta_results["moment"].append("before")
meta_results["type"].append("dev")
meta_results["loss"].append(dev_loss)
meta_results["acc"].append(dev_acc)
meta_results["time"].append(np.nan)


# # Train

# In[12]:


init = time()
global_step, tr_loss = train(train_dataset, model, tokenizer, hyperparams)
train_time = time() - init


# ### Eval after training
#
# #### Train sample

train_loss, train_results = evaluate(train_dataset_to_eval, hyperparams, model)
train_acc = (train_results.prediction == train_results.label).mean()

filtered = filter_df_by_label(df_train_to_eval.dropna()).reset_index(drop=True)
assert np.all(filtered.label.map(lambda x: lmap[x]) == train_results.label)

meta_results["moment"].append("after")
meta_results["type"].append("train")
meta_results["loss"].append(train_loss)
meta_results["acc"].append(train_acc)
meta_results["time"].append(train_time)


# #### Dev sample

dev_loss, results = evaluate(dev_dataset_to_eval, hyperparams, model)
dev_acc = (results.prediction == results.label).mean()


filtered = filter_df_by_label(df_dev_to_eval.dropna()).reset_index(drop=True)
assert np.all(filtered.label.map(lambda x: lmap[x]) == results.label)

meta_results["moment"].append("after")
meta_results["type"].append("dev")
meta_results["loss"].append(dev_loss)
meta_results["acc"].append(dev_acc)
meta_results["time"].append(train_time)


# ## Results

meta_results = pd.DataFrame(meta_results)
meta_results.to_csv("meta.csv", index=False)
