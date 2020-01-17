import numpy as np
import pandas as pd
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

from src.lr.text_processing.util import pre_process_nli_df  # noqa
from src.lr.text_processing.transformations.util import get_transformed_part_by_syn  # noqa
from src.lr.text_processing.transformations.util import syn2tranformation  # noqa
from src.lr.text_processing.transformations.util import get_augmented_data  # noqa
from src.lr.text_processing.transformations.synonyms import toy  # noqa
from src.lr.text_processing.transformations.structural import invert  # noqa
from src.lr.text_processing.transformations.structural import entailment_internalization  # noqa
from src.lr.training.util import label2ternary_label  # noqa

data_path = parentdir + "/src/data/toy/train.csv"
assert os.path.exists(data_path)


class BasicTransformation(unittest.TestCase):

    @classmethod
    def setUp(cls):
        df = pd.read_csv(data_path)
        pre_process_nli_df(df)
        cls.df = df

    def test_label(self):
        df_new = self.df.copy()         
        label2ternary_label(df_new)
        test = np.all(np.array([1, 0, -1]) == df_new.label.unique())
        self.assertTrue(test, "not correct label")

    def test_syn_transformation(self):
        df_t = get_transformed_part_by_syn(self.df, toy)
        aug1 = get_augmented_data(df=self.df,
                                  transformation=syn2tranformation(toy),
                                  frac=1)
        mods1 = ((aug1.premise != self.df.premise) | (
            aug1.hypothesis != self.df.hypothesis)).sum()
        self.assertEqual(
            mods1,
            df_t.shape[0],
            "not all transformations being done")
        self.assertEqual(aug1.shape, self.df.shape, "wrong shape")
        aug2 = get_augmented_data(df=self.df,
                                  transformation=syn2tranformation(toy),
                                  frac=0.5)
        mods2 = ((aug2.premise != self.df.premise) | (
            aug2.hypothesis != self.df.hypothesis)).sum()
        self.assertTrue(
            mods2 < mods1,
            "not the right number of transformations being done")
        self.assertEqual(aug2.shape, self.df.shape, "wrong shape")

        aug3 = get_augmented_data(df=self.df,
                                  transformation=syn2tranformation(toy),
                                  frac=0)
        mods3 = ((aug3.premise != self.df.premise) | (
            aug3.hypothesis != self.df.hypothesis)).sum()
        self.assertTrue(0 <= mods3 < mods2 < mods1,
                        "not the right number of transformations being done")
        self.assertEqual(aug3.shape, self.df.shape, "wrong shape")

    def test_invert_transformation(self):
        df_i = invert(self.df)
        aug = get_augmented_data(df=self.df, transformation=invert, frac=0.5)
        test1 = all(df_i.query("label!='entailment'").premise == self.df.query("label!='entailment'").hypothesis)  # noqa
        test2 = all(df_i.query("label!='entailment'").hypothesis == self.df.query("label!='entailment'").premise)  # noqa
        test3 = all(df_i.label == self.df.label)  # noqa
        test4 = (aug.query("label!='entailment'").hypothesis != self.df.query("label!='entailment'").hypothesis).max()  # noqa
        self.assertTrue(test1)
        self.assertTrue(test2)
        self.assertTrue(test3)
        self.assertTrue(test4)
        self.assertEqual(df_i.shape, self.df.shape)


    def test_entailment_internalization(self):
        df_i = entailment_internalization(self.df)
        aug = get_augmented_data(df=self.df, transformation=entailment_internalization, frac=0.5)
        test1 = all(df_i.label == self.df.label)  # noqa
        test2 = (aug.premise != self.df.premise).max()  # noqa
        test3 = (aug.hypothesis != self.df.hypothesis).max()  # noqa
        self.assertTrue(test1)
        self.assertTrue(test2)
        self.assertTrue(test3)
        self.assertEqual(df_i.premise.map(lambda x: len(x)).sum(), 0)
        self.assertEqual(df_i.shape, self.df.shape)

if __name__ == '__main__':
    unittest.main(verbosity=2)
