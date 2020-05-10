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

from src.lr.models.transformers.processor import filter_df_by_label, clean_df  # noqa
from src.lr.models.transformers.AlbertWrapper import AlbertWrapper  # noqa


base_path = parentdir + "/src/data/toy/"


class BasicAlbertTraining(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("example.log"):
            os.remove("example.log")

        if os.path.exists("albert"):
            shutil.rmtree("albert")

        paths = ["cached_dev_to_eval_200", "cached_test_200", "cached_train_200",
                 "cached_train_to_eval_200"]

        for path in paths:
            path = parentdir + path
            if os.path.exists(path):
                os.remove(path)

    def test_albert_training(self):

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
                       "max_steps": 7,
                       "warmup_steps": 0,
                       "save_steps": 6,
                       "no_cuda": False,
                       "n_gpu": 1,
                       "model_name_or_path": "albert",
                       "output_dir": "albert",
                       "random_state": 42,
                       "fp16": False,
                       "fp16_opt_level": "01",
                       "device": "cpu",
                       "verbose": False,
                       "model_type": "albert",
                       "pad_on_left": False,
                       "pad_token": 0,
                       "pad_token_segment_id": 0,
                       "mask_padding_with_zero": True,
                       "eval_sample_size": 100,
                       "n_cores": 7,
                       "base_path": base_path + "cached_",
                       "pretrained_weights": 'albert-base-v2'}

        # ## loading base model

        my_albert = AlbertWrapper(hyperparams)

        # ## Loading DFs

        train_path = base_path + "train.csv"
        test_path = base_path + "dev.csv"
        df = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test = test.sample(100, random_state=hyperparams["random_state"])
        df = clean_df(df, 7)
        test = clean_df(test, 7)

        # ### Eval 1

        pred = my_albert.predict(test, transform=True, mode="test")
        lmap = my_albert.processor.get_label_map()
        filtered = filter_df_by_label(test.dropna()).reset_index(drop=True)
        before_acc = np.mean(filtered.label.map(lambda x: lmap[x]) == pred)

        # ### Fit

        global_step, tr_loss, train_time = my_albert.fit(df)

        # ### Eval 2

        eval_path = base_path + "cached_test_200"
        pred = my_albert.predict(None, transform=False, path=eval_path)
        lmap = my_albert.processor.get_label_map()
        filtered = filter_df_by_label(test.dropna()).reset_index(drop=True)
        after_acc = np.mean(filtered.label.map(lambda x: lmap[x]) == pred)

        self.assertTrue(np.round(tr_loss, 3) == 1.153)
        self.assertTrue(before_acc == 0.32)
        self.assertTrue(after_acc == 0.37)


if __name__ == '__main__':
    unittest.main(verbosity=2)
