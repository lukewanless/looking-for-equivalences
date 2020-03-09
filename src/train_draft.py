#!/usr/bin/env python
# coding: utf-8

# In[16]:


from lr.models.transformers.util import *
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


# ### Params

# In[2]:


folder = "toy"

hyperparams = {"local_rank": -1,
               "max_seq_length": 200,
               "overwrite_cache": False,
               "num_train_epochs":3.0,
               "per_gpu_train_batch_size":32,
               "per_gpu_eval_batch_size":32,
               "gradient_accumulation_steps": 1,
               "learning_rate":5e-5,
               "weight_decay":0.0,
               "adam_epsilon": 1e-8,
               "max_grad_norm": 1.0,
               "max_steps": -1,
               "warmup_steps": 0,
               "save_steps": 500,
               "no_cuda":False,
               "n_gpu":1,
               "model_name_or_path":"bert",
               "output_dir":"bert",
               "random_state": 42,
               "fp16":False,
               "fp16_opt_level":"01",
               "device":"cpu",
               "verbose":True,
               "model_type": "bert",
               "train_cached_features_file": "data/{}/base_train_".format(folder),
               "dev_cached_features_file": "data/{}/base_dev_".format(folder)} 


set_seed(hyperparams["random_state"], hyperparams["n_gpu"])


# ## Set results dict

# In[3]:


meta_results = {"moment":[],
                "type":[],
                "loss":[],
                "acc":[],
                "time":[]}


# # df

# In[4]:


train_path = "data/{}/train.csv".format(folder)

df = pd.read_csv(train_path)
df_train, df_dev = train_test_split(df, test_size=0.1)


# # examples

# In[5]:


processor = NLIProcessor()
train_examples = processor.df2examples(df_train, "train")
dev_examples = processor.df2examples(df_dev, "dev")


# # features

# In[6]:


pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
label_map = processor.get_label_map()
max_seq_length = hyperparams["max_seq_length"]

train_cached_features_file = hyperparams["train_cached_features_file"]
dev_cached_features_file = hyperparams["dev_cached_features_file"]



train_features = convert_examples_to_features(examples=train_examples,
                                              tokenizer=tokenizer,
                                              label_map=label_map,
                                              max_length=max_seq_length)


dev_features = convert_examples_to_features(examples=dev_examples,
                                              tokenizer=tokenizer,
                                              label_map=label_map,
                                              max_length=max_seq_length)

torch.save(train_features, train_cached_features_file)

torch.save(dev_features, dev_cached_features_file)


# # dataset

# In[7]:


train_dataset = features2dataset(train_cached_features_file, hyperparams, evaluate=False)
dev_dataset = features2dataset(dev_cached_features_file, hyperparams, evaluate=True)


# ## Loading Model

# In[8]:


model = BertForSequenceClassification.from_pretrained(pretrained_weights, num_labels = 3)


# ### Eval before training

# #### train

# In[9]:


train_loss, train_results = evaluate(train_dataset, hyperparams, model)
train_acc = (train_results.prediction==train_results.label).mean()

meta_results["moment"].append("before")
meta_results["type"].append("train")
meta_results["loss"].append(train_loss)
meta_results["acc"].append(train_acc)
meta_results["time"].append(np.nan)


# #### Dev 

# In[10]:


dev_loss, results = evaluate(dev_dataset, hyperparams, model)
dev_acc = (results.prediction==results.label).mean()


meta_results["moment"].append("before")
meta_results["type"].append("dev")
meta_results["loss"].append(dev_loss)
meta_results["acc"].append(dev_acc)
meta_results["time"].append(np.nan)


# # Train

# In[11]:


init = time()
global_step, tr_loss = train(train_dataset, model, tokenizer, hyperparams)
train_time = time() - init


# ### Eval After training

# #### train

# In[13]:


train_loss, train_results = evaluate(train_dataset, hyperparams, model)
train_acc = (train_results.prediction==train_results.label).mean()

meta_results["moment"].append("after")
meta_results["type"].append("train")
meta_results["loss"].append(train_loss)
meta_results["acc"].append(train_acc)
meta_results["time"].append(train_time)


# #### dev

# In[14]:


dev_loss, results = evaluate(dev_dataset, hyperparams, model)
dev_acc = (results.prediction==results.label).mean()

meta_results["moment"].append("after")
meta_results["type"].append("dev")
meta_results["loss"].append(dev_loss)
meta_results["acc"].append(dev_acc)
meta_results["time"].append(train_time)


# ## Save results

# In[15]:


meta_results = pd.DataFrame(meta_results)
meta_results.to_csv("meta.csv",index=False)


# In[ ]:




