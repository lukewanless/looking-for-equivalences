import numpy as np
import pandas as pd
from time import time
import shutil
import os
import sys
import inspect
import unittest

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from src.lr.models.transformers.processor import clean_df  # noqa
from src.lr.models.transformers.train_functions import set_seed  # noqa
from src.lr.models.transformers.BertWrapper import BertWrapper  # noqa
from src.lr.text_processing.transformations.structural import entailment_internalization  # noqa
from src.lr.stats.h_testing import *  # noqa


base_path = parentdir + "/src/data/toy/"


class BertHTesting(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("example.log"):
            os.remove("example.log")

        if os.path.exists("bert_draft"):
            shutil.rmtree("bert_draft")

        paths = ["cached_dev_to_eval_200", "cached_test_200",
                 "cached_test_t_200", "cached_train_200",
                 "cached_train_to_eval_200"]

        for path in paths:
            path = parentdir + path
            if os.path.exists(path):
                os.remove(path)

    def test_bert_h_testing(self):

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
                       "max_steps": 4,
                       "warmup_steps": 0,
                       "save_steps": 3,
                       "no_cuda": False,
                       "n_gpu": 1,
                       "data_set_name": "toy",
                       "transformation_name": "entailment internalization",
                       "rho": 0.3,
                       "model_name_or_path": "bert",
                       "output_dir": "bert_draft",
                       "random_state": 42,
                       "dgp_seed": 42,
                       "fp16": False,
                       "fp16_opt_level": "01",
                       "device": "cpu",
                       "verbose": False,
                       "model_type": "bert",
                       "pad_on_left": False,
                       "pad_token": 0,
                       "n_cores": 7,
                       'eval_sample_size': 100,
                       "pad_token_segment_id": 0,
                       "mask_padding_with_zero": True,
                       "base_path": base_path + "cached_"}

        # ## Loading DFs

        train_path = base_path + "train.csv"
        test_path = base_path + "dev.csv"

        transformation = entailment_internalization

        train = pd.read_csv(train_path)
        dev_o = pd.read_csv(test_path)
        dev_t = transformation(dev_o)

        train = clean_df(train, n_cores=8)
        dev_o = clean_df(dev_o, n_cores=8)
        dev_t = clean_df(dev_t, n_cores=8)


        set_seed(hyperparams["random_state"], 0)
        dgp = DGP(train, transformation, rho=0.3)
        train_ = dgp.sample()


        # ### Test

        test_results =  h_test_transformer(df_train=train_,
                                           df_dev=dev_o,
                                           df_dev_t=dev_t,
                                           ModelWrapper=BertWrapper,
                                           hyperparams=hyperparams,
                                           S=1000)

        # ### Fit

        self.assertTrue(test_results.observable_t_stats[0] == 0.5985858317644218)
        self.assertTrue(test_results.validation_accuracy[0] == 0.36)
        self.assertTrue(test_results.transformed_validation_accuracy[0] == 0.325)
        self.assertTrue(test_results.p_value[0] == 0.604)

        p_sum = (test_results[[c for c in test_results.columns if c.find("boot") >-1]]).sum(1)[0] 
        self.assertTrue(p_sum == 21.92981436880456)

if __name__ == '__main__':
    unittest.main(verbosity=2)


