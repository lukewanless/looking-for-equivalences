from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import sys
import inspect
import unittest
import shutil

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir+"/src")

from search_xgb import search

train_path = parentdir + "/src/data/snli/train_sample.csv"
dev_path = parentdir + "/src/data/snli/dev.csv"


class XGBsearch(unittest.TestCase):
    
    @classmethod
    def setUp(cls):
        folder = "search_draft"
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    @classmethod
    def tearDown(cls):
        if os.path.exists("search_draft"):
            shutil.rmtree("search_draft")

    def test_search(self):
        search(train_path=train_path,
               dev_path=dev_path,
               random_state=223,
               cv=5,
               n_iter=5,
               n_cores=8,
               output_dir_name="search_draft",
               verbose=False)
        
        result = pd.read_csv("search_draft/search_223.csv")
        eval_ = result.loc[0, "expected_val_score"]
        bval_ = result.loc[0, "best_val_score"]
        self.assertTrue(eval_ == 0.4915374136432657)
        self.assertTrue(bval_ == 0.5225563909774437)


if __name__ == '__main__':
    unittest.main(verbosity=2)
