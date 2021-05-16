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


# def trans(df):
#     return p_h_transformation_syn_dict(df, syn_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('n_cores',
                        type=int,
                        help='number of cores')
    args = parser.parse_args()


    # variables
    n_cores = args.n_cores

    train_path = "data/sick/train.csv"
    test_path = "data/sick/test.csv"

    veto_path = "data/sick/syn_veto.csv"
    syn_path = "data/sick/syn_noun.csv"

    output_train_path = "data/sick/train_p_h_syn_noun.csv"
    output_test_path = "data/sick/test_p_h_syn_noun.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # cleaning
    train = filter_df_by_label(train.dropna()).reset_index(drop=True)
    test = filter_df_by_label(test.dropna()).reset_index(drop=True)


    pre_process_nli_df(train)
    pre_process_nli_df(test)




    print("train.shape", train.shape)
    print("test.shape", test.shape)

    # ## Get syn dict

    # corpus = train + dev
    df_corpus = pd.concat([train])

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

    init = time()
    train_t = parallelize(df=train, func=p_h_transformation_syn_dict, n_cores=n_cores, syn_dictp=syn_dict)
    trans_time = time() - init
    print("applying trans to train: {:.4f} minutes".format(trans_time / 60))

    init = time()
    test_t = parallelize(df=test, func=p_h_transformation_syn_dict, n_cores=n_cores, syn_dictp=syn_dict)
    trans_time = time() - init
    print("applying trans to test: {:.4f} minutes".format(trans_time / 60))

    # return clean
    train.to_csv('data/sick/train_clean.csv', index=False)
    test.to_csv('data/sick/test_clean.csv', index=False)

    for index, row in train_t.iterrows():
        row['premise'] = row['premise'].strip()
        row['hypothesis'] = row['hypothesis'].strip()

    train_t.to_csv(output_train_path, index=False)

    for index, row in test_t.iterrows():
        row['premise'] = row['premise'].strip()
        row['hypothesis'] = row['hypothesis'].strip()

    test_t.to_csv(output_test_path, index=False)
