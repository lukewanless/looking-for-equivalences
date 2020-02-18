import pandas as pd
import numpy as np
from time import time
from lr.text_processing.util import pre_process_nli_df
from lr.text_processing.util import get_corpus
from lr.text_processing.transformations.wordnet import get_noun_syn_dict
from lr.text_processing.transformations.wordnet import p_h_transformation_syn_dict
from lr.text_processing.transformations.wordnet import parallelize
from lr.training.util import filter_df_by_label

debug = False
n_cores = 8

# ### Loading data
train_path = "data/snli/train.csv"
dev_path = "data/snli/dev.csv"

train_path_mod = "data/snli/train_p_h_syn_noun.csv"
dev_path_mod = "data/snli/dev_p_h_syn_noun.csv"

train = pd.read_csv(train_path)
dev = pd.read_csv(dev_path)

if debug:
    train = train.head(1000)
    dev = dev.head(1000)

train = filter_df_by_label(train.dropna()).reset_index(drop=True)
dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
pre_process_nli_df(train)
pre_process_nli_df(dev)


# get syn dict

init = time()
syn_dict = get_noun_syn_dict(df=train, n_cores=n_cores)
syn_time = time() - init
print("get syn dict: {:.4f} minutes".format(syn_time / 60))



# apply transformation on the whole dataset
def trans(df):
    return p_h_transformation_syn_dict(df, syn_dict)  



init = time()
train_t = parallelize(df=train,func=trans, n_cores=n_cores)
trans_time = time() - init
print("applying trans to train: {:.4f} minutes".format(trans_time / 60))


init = time()
dev_t = parallelize(df=dev, func=trans, n_cores=n_cores)
trans_time = time() - init
print("applying trans to train: {:.4f} minutes".format(trans_time / 60))


train_t.to_csv(train_path_mod, index=False)
dev_t.to_csv(dev_path_mod, index=False)

