import pandas as pd
import numpy as np
from time import time
from lr.text_processing.util import pre_process_nli_df
from lr.text_processing.transformations.wordnetsyn import p_h_transformation_noun_minimal_edition
from lr.text_processing.transformations.wordnetsyn import parallelize, path_base_transormation
from lr.training.util import filter_df_by_label

debug = True

if debug:
    folder = "toy"
    n_cores = 2

else:
    folder = "snli"
    n_cores = 6    


train_path = "data/{}/train.csv".format(folder)
dev_path = "data/{}/dev.csv".format(folder)

train_path_mod = "data/{}/train_mod.csv".format(folder)
dev_path_mod = "data/{}/dev_mod.csv".format(folder)

syn_transformation = p_h_transformation_noun_minimal_edition

train = pd.read_csv(train_path)
dev = pd.read_csv(dev_path)
train = filter_df_by_label(train.dropna()).reset_index(drop=True)
dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
pre_process_nli_df(train)
pre_process_nli_df(dev)



# ### Transforming data


def transformation(df):
    return parallelize(df,
                       func=syn_transformation,
                       n_cores=n_cores)


print("transforming dev....\n")
init = time()
dev_t = transformation(dev)
d_time = time() - init
print("dev tranformed time = {:.3f} minutes\n".format(d_time / 60))


print("transforming train....\n")
init = time()
train_t = transformation(train)
t_time = time() - init
print("train tranformed time = {:.3f} minutes\n".format(t_time / 60))


# Saving transformations

dev_t.to_csv(dev_path_mod, index=False)
train_t.to_csv(train_path_mod, index=False)
