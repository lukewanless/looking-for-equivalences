import numpy as np
import pandas as pd
import os
import sys
import inspect
import unittest
from joblib import load


currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.lr.text_processing.util import pre_process_nli_df
from src.lr.text_processing.util import get_corpus
from src.lr.text_processing.transformations.wordnet import get_noun_syn_dict
from src.lr.text_processing.transformations.wordnet import p_h_transformation_syn_dict
from src.lr.text_processing.transformations.wordnet import parallelize
from src.lr.training.util import filter_df_by_label


# data_path = parentdir + "/src/data/toy/train.csv"
# assert os.path.exists(data_path)


class SynTrans(unittest.TestCase):
    @classmethod
    def setUp(cls):
        folder = "toy"
        cls.n_cores = 2
        train_path = parentdir + "/src/data/{}/train.csv".format(folder)
        dev_path = parentdir + "/src/data/{}/dev.csv".format(folder)

        veto_path = parentdir + "/src/data/{}/syn_veto.csv".format(folder)
        cls.syn_path = parentdir + "/src/data/{}/syn_noun.csv".format(folder)

        cls.train_path_mod = parentdir + "/src/data/{}/train_p_h_syn_noun.csv".format(folder)
        cls.dev_path_mod = parentdir + "/src/data/{}/dev_p_h_syn_noun.csv".format(folder)

        train = pd.read_csv(train_path)
        dev = pd.read_csv(dev_path)


        train = filter_df_by_label(train.dropna()).reset_index(drop=True)
        dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)
        pre_process_nli_df(train)
        pre_process_nli_df(dev)
        cls.train = train 
        cls.dev = dev
        cls.veto = pd.read_csv(veto_path).veto.values


    @classmethod
    def tearDown(cls):
      for path in [cls.train_path_mod, cls.dev_path_mod, cls.syn_path]:
          if os.path.exists(path):
                os.remove(path)

    def test_syn_transformation(self):

        # get syn dict
        syn_dict = get_noun_syn_dict(df=self.train, n_cores=self.n_cores, veto=self.veto)
        # removing possible verbs
        syn_dict = {k: syn_dict[k] for k in syn_dict if k[-3:] != "ing"}
        # saving to a dataframe
        key = sorted(syn_dict.keys())
        value = [syn_dict[k] for k in key]

        syn_df = pd.DataFrame({"key": key,
                               "value": value})

        train_t = p_h_transformation_syn_dict(df=self.train, syn_dict=syn_dict)
        dev_t = p_h_transformation_syn_dict(df=self.dev, syn_dict=syn_dict)


        self.assertTrue(not np.any(self.train.premise == train_t.premise))
        self.assertTrue(not np.any(self.train.hypothesis == train_t.hypothesis))
        self.assertTrue(np.all(self.train.label == train_t.label))


        self.assertTrue(not np.any(self.dev.premise == dev_t.premise))
        self.assertTrue(not np.any(self.dev.hypothesis == dev_t.hypothesis))
        self.assertTrue(np.all(self.dev.label == dev_t.label))

 


if __name__ == '__main__':
    unittest.main(verbosity=2)
