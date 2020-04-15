import pandas as pd
import numpy as np
from time import time
from lr.text_processing.util import pre_process_nli_df
from lr.text_processing.util import get_corpus
from lr.text_processing.transformations.wordnet import get_noun_syn_dict
from lr.text_processing.transformations.wordnet import p_h_transformation_syn_dict
from lr.text_processing.transformations.wordnet import parallelize
from lr.training.util import filter_df_by_label
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('folder',
                    type=str,
                    help='data folder')

parser.add_argument('n_cores',
                    type=int,
                    help='number of cores')
args = parser.parse_args()


# variables
folder = args.folder
n_cores = args.n_cores

train_path = "data/{}/train.csv".format(folder)
train_sample_path = "data/{}/train_sample.csv".format(folder)
train_not_in_sample_path = "data/{}/train_not_in_sample.csv".format(folder)

dev_path = "data/{}/dev.csv".format(folder)
test_path = "data/{}/test.csv".format(folder)

veto_path = "data/{}/syn_veto.csv".format(folder)
syn_path = "data/{}/syn_noun.csv".format(folder)

output_train_path = "data/{}/train_p_h_syn_noun.csv".format(folder)
output_train_sample_path = "data/{}/train_sample_p_h_syn_noun.csv".format(
    folder)

output_train_not_in_sample_path = "data/{}/train_not_in_sample_p_h_syn_noun.csv".format(
    folder)
output_dev_path = "data/{}/dev_p_h_syn_noun.csv".format(folder)
output_test_path = "data/{}/test_p_h_syn_noun.csv".format(folder)


train = pd.read_csv(train_path)
train_sample = pd.read_csv(train_sample_path)
train_not_in_sample = pd.read_csv(train_not_in_sample)

dev = pd.read_csv(dev_path)
test = pd.read_csv(test_path)

# cleaning
train = filter_df_by_label(train.dropna()).reset_index(drop=True)
train_sample = filter_df_by_label(train_sample.dropna()).reset_index(drop=True)
train_not_in_sample = filter_df_by_label(
    train_not_in_sample.dropna()).reset_index(
        drop=True)

dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
test = filter_df_by_label(test.dropna()).reset_index(drop=True)

pre_process_nli_df(train)
pre_process_nli_df(train_sample)
pre_process_nli_df(train_not_in_sample)
pre_process_nli_df(dev)
pre_process_nli_df(test)

print("train.shape", train.shape)
print("train_sample.shape", train_sample.shape)
print("train_not_in_sample.shape", train_not_in_sample.shape)
print("dev.shape", dev.shape)
print("test.shape", test.shape)

# ## Get syn dict

# corpus = train + dev
df_corpus = pd.concat([train, dev])

# defining words that will no be used
veto = pd.read_csv(veto_path).veto.values

# get syn dict
init = time()
syn_dict = get_noun_syn_dict(df=df_corpus, n_cores=n_cores, veto=veto)
del df_corpus
# removing possible verbs
syn_dict = {k: syn_dict[k] for k in syn_dict if k[-3:] != "ing"}

# saving to a dataframe
key = sorted(syn_dict.keys())
value = [syn_dict[k] for k in key]
syn_df = pd.DataFrame({"key": key,
                       "value": value})

syn_df.to_csv(syn_path, index=False)

syn_time = time() - init
print("get syn dict: {:.4f} minutes".format(syn_time / 60))

# ## Apply transformation on the whole dataset


def trans(df):
    return p_h_transformation_syn_dict(df, syn_dict)


init = time()
train_t = parallelize(df=train, func=trans, n_cores=n_cores)
trans_time = time() - init
print("applying trans to train: {:.4f} minutes".format(trans_time / 60))

init = time()
train_sample_t = parallelize(df=train_sample, func=trans, n_cores=n_cores)
trans_time = time() - init
print("applying trans to train_sample: {:.4f} minutes".format(trans_time / 60))


init = time()
train_not_in_sample_t = parallelize(
    df=train_not_in_sample,
    func=trans,
    n_cores=n_cores)
trans_time = time() - init
print(
    "applying trans to train_not_in_sample: {:.4f} minutes".format(
        trans_time /
        60))


init = time()
dev_t = parallelize(df=dev, func=trans, n_cores=n_cores)
trans_time = time() - init
print("applying trans to dev: {:.4f} minutes".format(trans_time / 60))

init = time()
test_t = parallelize(df=test, func=trans, n_cores=n_cores)
trans_time = time() - init
print("applying trans to test: {:.4f} minutes".format(trans_time / 60))

train_t.to_csv(output_train_path, index=False)
train_sample_t.to_csv(output_train_sample_path, index=False)
train_not_in_sample.to_csv(output_train_not_in_sample_path, index=False)
dev_t.to_csv(output_dev_path, index=False)
test_t.to_csv(output_test_path, index=False)
