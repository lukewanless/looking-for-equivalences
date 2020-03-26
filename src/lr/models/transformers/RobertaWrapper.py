import logging
import os
import shutil
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from time import time
from sklearn.model_selection import train_test_split

try:
    from lr.models.transformers.processor import NLIProcessor, features2dataset  # noqa
    from lr.models.transformers.processor import filter_df_by_label  # noqa
    from lr.models.transformers.train_functions import evaluate, train, set_seed  # noqa

except ModuleNotFoundError:
    from src.lr.models.transformers.processor import NLIProcessor, features2dataset  # noqa
    from src.lr.models.transformers.processor import filter_df_by_label  # noqa
    from src.lr.models.transformers.train_functions import evaluate, train, set_seed  # noqa


class RobertaWrapper():
    """
    Roberta wrapper
    """

    def __init__(self,
                 hyperparams,
                 pretrained_weights='roberta-base'):
        """
        :param hyperparams: list of paranters
        :type hyperparams: dict

        pretrained_weights in ['roberta-base',
                               'roberta-large',
                               'roberta-large-mnli',
                               'distilroberta-base',
                               'roberta-base-openai-detector',
                               'roberta-large-openai-detector']
        """
        set_seed(hyperparams["random_state"], hyperparams["n_gpu"])

        pretrained_weights = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        hyperparams["tokenizer"] = self.tokenizer
        self.hyperparams = hyperparams
        self.model = RobertaForSequenceClassification.from_pretrained(
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

        train_cached_features_file = self.transform(df_train,
                                                    mode="train")

        train_to_eval_cached_features_file = self.transform(df_train_to_eval,
                                                            mode="train_to_eval")

        dev_to_eval_cached_features_file = self.transform(df_dev_to_eval,
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

    def full_predict(self, df, transform, mode, path):
        n_cores = self.hyperparams["n_cores"]
        verbose = self.hyperparams["verbose"]

        if transform:
            eval_cached_features_file = self.transform(df, mode)

            if verbose:
                print("eval path = ", eval_cached_features_file)

        else:
            eval_cached_features_file = path

        eval_dataset = features2dataset(eval_cached_features_file)
        _, results = evaluate(eval_dataset, self.hyperparams, self.model)
        return results

    def predict(self, df, transform=True, mode="eval", path=None):
        results = self.full_predict(df, transform, mode=mode, path=path)
        return results.prediction

    def load(self, path):
        self.model = RobertaForSequenceClassification.from_pretrained(path)

    def transform(self, df, mode):
        n_cores = self.hyperparams["n_cores"]
        eval_cached_features_file = self.processor.df2features(df=df,
                                                               n_cores=n_cores,
                                                               mode=mode)
        return eval_cached_features_file

    def get_results(self, df, mode):
        results = self.full_predict(df, transform=True, mode=mode, path=None)
        lmap = self.processor.get_label_map()
        filtered = filter_df_by_label(df.dropna()).reset_index(drop=True)
        assert np.all(filtered.label.map(lambda x: lmap[x]) == results.label)
        results.loc[:, "indicator"] = results.label == results.prediction
        results.loc[:, "indicator"] = results.indicator.apply(lambda x: int(x))
        return results
    
    def get_param_count(self):
        params = list(self.model.parameters())
        pp=0
        for p in params:
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
