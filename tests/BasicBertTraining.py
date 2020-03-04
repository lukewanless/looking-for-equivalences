import numpy as np
import pandas as pd
import os
import shutil
import torch
import os
import sys
import inspect
import unittest
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification


currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from src.lr.models.transformers.util import load_and_cache_examples  # noqa
from src.lr.models.transformers.util import train, set_seed  # noqa


train_path = parentdir + "/src/data/toy/train.csv"
dev_path = parentdir + "/src/data/toy/dev.csv"
cache_path = parentdir + "/src/data/toy/"



class BasicBertTraining(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("example.log"):
            os.remove("example.log")
        if os.path.exists("bert"):
            shutil.rmtree("bert")

    def test_bert_training(self):
        hyperparams = {"local_rank": -1,
                       "max_seq_length": 128,
                       "overwrite_cache": False,
                       "cached_path": cache_path,
                       "train_path": train_path,
                       "dev_path": dev_path,
                       "num_train_epochs": 3.0,
                       "per_gpu_train_batch_size": 8,
                       "per_gpu_eval_batch_size": 8,
                       "gradient_accumulation_steps": 1,
                       "learning_rate": 5e-5,
                       "weight_decay": 0.0,
                       "adam_epsilon": 1e-8,
                       "max_grad_norm": 1.0,
                       "max_steps": 10,
                       "warmup_steps": 0,
                       "save_steps": 5,
                       "no_cuda": True,
                       "n_gpu": 1,
                       "model_name_or_path": "bert",
                       "output_dir": "bert",
                       "random_state": 42,
                       "fp16": False,
                       "fp16_opt_level": "01",
                       "device": "cpu",
                       "verbose": False,
                       "model_type": "bert"}

        set_seed(hyperparams["random_state"], hyperparams["n_gpu"])

        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = BertForSequenceClassification.from_pretrained(
            pretrained_weights, num_labels=3)

        train_dataset = load_and_cache_examples(hyperparams, tokenizer)
        dev_dataset = load_and_cache_examples(
            hyperparams, tokenizer, evaluate=True)
        global_step, tr_loss = train(
            train_dataset, model, tokenizer, hyperparams)
        training_logs = pd.read_csv("bert/log.csv")
        a1 = training_logs.loss.rolling(3).mean().iloc[3]
        a2 = training_logs.loss.rolling(3).mean().iloc[-1]
        self.assertTrue(a1 > a2)
        self.assertTrue(a1 == 1.3554697434107463)
        self.assertTrue(a2 == 1.131113092104594)
        self.assertTrue(tr_loss == 1.195960673418912)


if __name__ == '__main__':
    unittest.main(verbosity=2)
