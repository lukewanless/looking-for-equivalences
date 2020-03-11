from lr.models.transformers.processor import NLIProcessor, features2dataset
from lr.models.transformers.processor import filter_df_by_label
from lr.models.transformers.train_functions import evaluate, train, set_seed
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


class BertWrapper():
    """
    Bert wrapper
    """

    def __init__(self, hyperparams):
        """
        :param hyperparams: list of paranters
        :type hyperparams: dict
        """
        set_seed(hyperparams["random_state"], hyperparams["n_gpu"])

        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        hyperparams["tokenizer"] = self.tokenizer
        self.hyperparams = hyperparams
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_weights, num_labels=3)
        self.processor = NLIProcessor(hyperparams)

    def fit(self, df):
        n_cores = self.hyperparams["n_cores"]
        eval_sample_size = self.hyperparams["eval_sample_size"]
        random_state = self.hyperparams["random_state"]

        df_train, df_dev_to_eval = train_test_split(
            df, test_size=eval_sample_size)
        df_train_to_eval = df_train.sample(
            n=eval_sample_size,
            random_state=random_state)

        train_cached_features_file = self.processor.df2features(df=df_train,
                                                                n_cores=n_cores,
                                                                mode="train")

        train_to_eval_cached_features_file = self.processor.df2features(df=df_train_to_eval,
                                                                        n_cores=n_cores,
                                                                        mode="train_to_eval")

        dev_to_eval_cached_features_file = self.processor.df2features(df=df_dev_to_eval,
                                                                      n_cores=n_cores,
                                                                      mode="dev_to_eval")

        del df_train, df_train_to_eval
        del df_dev_to_eval

        train_dataset = features2dataset(train_cached_features_file)
        train_dataset_to_eval = features2dataset(
            train_to_eval_cached_features_file)
        dev_dataset_to_eval = features2dataset(
            dev_to_eval_cached_features_file)

        init = time()
        global_step, tr_loss = train(train_dataset,
                                     train_dataset_to_eval,
                                     dev_dataset_to_eval,
                                     self.model,
                                     self.tokenizer,
                                     self.hyperparams)
        train_time = time() - init
        return global_step, tr_loss, train_time

    def predict(self, df, transform=True, mode="eval", path=None):
        n_cores = self.hyperparams["n_cores"]
        verbose = self.hyperparams["verbose"]
        if transform:
            eval_cached_features_file = self.processor.df2features(df=df,
                                                                   n_cores=n_cores,
                                                                   mode=mode)
            if verbose:
                print("eval path = ", eval_cached_features_file)

        else:
            eval_cached_features_file = path

        eval_dataset = features2dataset(eval_cached_features_file)
        _, results = evaluate(eval_dataset, self.hyperparams, self.model)
        return results.prediction
