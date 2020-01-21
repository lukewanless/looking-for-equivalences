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

from src.lr.text_processing.util import pre_process_nli_df  # noqa
from src.lr.training.util import get_binary_label, get_ternary_label  # noqa
from src.lr.models.logistic_regression import LRWrapper  # noqa
from src.lr.training.language_representation import BOW, Tfidf  # noqa


data_path = parentdir + "/src/data/toy/train.csv"
assert os.path.exists(data_path)


class BasicLrTraining(unittest.TestCase):
    @classmethod
    def setUp(cls):
        df = pd.read_csv(data_path)
        pre_process_nli_df(df)
        cls.df = df
        cls.path = "test.joblib"

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.path):
            os.remove(cls.path)

    def test_train_binary_BOW(self):
        hyperparams = {"RepresentationFunction": BOW,
                       "num_words": 500,
                       "label_translation": get_binary_label,
                       "penalty": "l2",
                       "C": 1,
                       'solver': 'lbfgs'}
        lr = LRWrapper(hyperparams)
        lr.fit(self.df)
        acc = lr.get_acc(self.df)
        baseline = (
            (self.df.label.value_counts() /
             self.df.shape[0])[
                1:]).sum()
        msg = "acc = {:.2f} baseline  = {:.2f}".format(acc, baseline)
        self.assertTrue(acc > baseline, msg)

    def test_train_ternary_Tfidf(self):
        hyperparams = {"RepresentationFunction": Tfidf,
                       "max_features": 500,
                       "label_translation": get_ternary_label,
                       "penalty": "l2",
                       "C": 1,
                       'solver': 'lbfgs'}
        lr = LRWrapper(hyperparams)
        lr.fit(self.df)
        acc = lr.get_acc(self.df)
        baseline = 0.4
        msg = "acc = {:.2f} baseline  = {:.2f}".format(acc, baseline)
        self.assertTrue(acc > baseline, msg)



    def test_save(self):
        hyperparams = {"RepresentationFunction": Tfidf,
                       "max_features": 500,
                       "label_translation": get_ternary_label,
                       "penalty": "l2",
                       "C": 1,
                       'solver': 'lbfgs'}
        lr = LRWrapper(hyperparams)
        lr.fit(self.df)
        scores = lr.get_score(self.df)
        lr.save(self.path)
        del lr
        lr = load(self.path)
        new_scores = lr.get_score(self.df)
        msg = "problem to save results"
        self.assertTrue(all(scores == new_scores), msg)
    #     df_new = self.df.copy()
    #     label2ternary_label(df_new)
    #     test = np.all(np.array([1, 0, -1]) == df_new.label.unique())
    #     self.assertTrue(test, "not correct label")

    # def test_syn_transformation(self):
    #     df_t = get_transformed_part_by_syn(self.df, toy)
    #     aug1 = get_augmented_data(df=self.df,
    #                               transformation=syn2tranformation(toy),
    #                               frac=1)
    #     mods1 = ((aug1.premise != self.df.premise) | (
    #         aug1.hypothesis != self.df.hypothesis)).sum()
    #     self.assertEqual(
    #         mods1,
    #         df_t.shape[0],
    #         "not all transformations being done")
    #     self.assertEqual(aug1.shape, self.df.shape, "wrong shape")
    #     aug2 = get_augmented_data(df=self.df,
    #                               transformation=syn2tranformation(toy),
    #                               frac=0.5)
    #     mods2 = ((aug2.premise != self.df.premise) | (
    #         aug2.hypothesis != self.df.hypothesis)).sum()
    #     self.assertTrue(
    #         mods2 < mods1,
    #         "not the right number of transformations being done")
    #     self.assertEqual(aug2.shape, self.df.shape, "wrong shape")

    #     aug3 = get_augmented_data(df=self.df,
    #                               transformation=syn2tranformation(toy),
    #                               frac=0)
    #     mods3 = ((aug3.premise != self.df.premise) | (
    #         aug3.hypothesis != self.df.hypothesis)).sum()
    #     self.assertTrue(0 <= mods3 < mods2 < mods1,
    #                     "not the right number of transformations being done")
    #     self.assertEqual(aug3.shape, self.df.shape, "wrong shape")

    # def test_invert_transformation(self):
    #     df_i = invert(self.df)
    #     aug = get_augmented_data(df=self.df, transformation=invert, frac=0.5)
    #     test1 = all(df_i.query("label!='entailment'").premise == self.df.query("label!='entailment'").hypothesis)  # noqa
    #     test2 = all(df_i.query("label!='entailment'").hypothesis == self.df.query("label!='entailment'").premise)  # noqa
    #     test3 = all(df_i.label == self.df.label)  # noqa
    #     test4 = (aug.query("label!='entailment'").hypothesis != self.df.query("label!='entailment'").hypothesis).max()  # noqa
    #     self.assertTrue(test1)
    #     self.assertTrue(test2)
    #     self.assertTrue(test3)
    #     self.assertTrue(test4)
    #     self.assertEqual(df_i.shape, self.df.shape)

    # def test_entailment_internalization(self):
    #     df_i = entailment_internalization(self.df)
    #     aug = get_augmented_data(df=self.df, transformation=entailment_internalization, frac=0.5)
    #     test1 = all(df_i.label == self.df.label)  # noqa
    #     test2 = (aug.premise != self.df.premise).max()  # noqa
    #     test3 = (aug.hypothesis != self.df.hypothesis).max()  # noqa
    #     self.assertTrue(test1)
    #     self.assertTrue(test2)
    #     self.assertTrue(test3)
    #     self.assertEqual(df_i.premise.map(lambda x: len(x)).sum(), 0)
    #     self.assertEqual(df_i.shape, self.df.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
