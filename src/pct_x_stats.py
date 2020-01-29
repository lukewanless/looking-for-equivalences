import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce
from lr.text_processing.util import pre_process_nli_df, get_vocab_count
from lr.text_processing.transformations.util import syn_h2tranformation, syn_p2tranformation
from lr.text_processing.transformations.util import syn2tranformation
from lr.text_processing.transformations.util import get_augmented_data
from lr.text_processing.transformations.structural import invert, entailment_internalization
from lr.training.util import get_binary_label, get_ternary_label
from lr.training.util import filter_df_by_label
from lr.training.language_representation import Tfidf
from lr.models.logistic_regression import LRWrapper
from lr.stats.matched_comparison import get_disagreement_statistics
from lr.stats.bootstrap import get_bootstrap_replications, apply_transformation_under_H0
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# # load data

train_path = "data/snli/train.csv"
dev_path = "data/snli/dev.csv"


debug = False


result_path = "results/snli_lr_Tfidf_ent_int_30_40.csv"
ent_result_path = "results/snli_lr_Tfidf_ent_int_ent_30_40.csv"
neutral_result_path = "results/snli_lr_Tfidf_ent_int_neutral_30_40.csv"
contra_result_path = "results/snli_lr_Tfidf_ent_int_contra_30_40.csv"

if debug:

    result_path = "results/t.csv"
    ent_result_path = "results/t.csv"
    neutral_result_path = "results/t.csv"
    contra_result_path = "results/t.csv"


df = pd.read_csv(train_path)
dev = pd.read_csv(dev_path)

if debug:
    df = df.head(100)
    dev = dev.head(100)

df = filter_df_by_label(df.dropna()).reset_index(drop=True)
dev = filter_df_by_label(dev.dropna()).reset_index(drop=True)

pre_process_nli_df(df)
pre_process_nli_df(dev)

trials = 30
B = 40
max_features = None

if debug:
    trials = 3
    B = 10
    max_features = 500

label_translation = get_ternary_label
transformation = entailment_internalization


def trans_df(x): return get_augmented_data(df=x,
                                           transformation=transformation,
                                           frac=1)


pcts = np.linspace(0, 1, trials)
ids = reduce(lambda x, y: x + y, [[i] * B for i in pcts])

stats = []
ent_stats = []
contra_stats = []
neutral_stats = []

hyperparams = {"RepresentationFunction": Tfidf,
               "max_features": max_features,
               "label_translation": label_translation,
               "penalty": "l2",
               "C": 1,
               'solver': 'lbfgs'}

for pct in tqdm(ids):

    train_b = get_bootstrap_replications(df)
    dev_b = get_bootstrap_replications(dev)

    train_b_aug = get_augmented_data(df=df,
                                     transformation=transformation,
                                     frac=pct)

    dev_b_original, dev_b_t = apply_transformation_under_H0(
        dev_b, df_transformation_f=trans_df)

    lr = LRWrapper(hyperparams)
    lr.fit(train_b_aug)

    st = get_disagreement_statistics(df=dev_b_original,
                                     df_t=dev_b_t,
                                     model=lr,
                                     label_translation=label_translation)

    neutral_st = get_disagreement_statistics(df=dev_b_original.query("label=='neutral'"),
                                             df_t=dev_b_t.query(
                                                 "label=='neutral'"),
                                             model=lr,
                                             label_translation=label_translation)

    contra_st = get_disagreement_statistics(df=dev_b_original.query("label=='contradiction'"),
                                            df_t=dev_b_t.query(
                                                "label=='contradiction'"),
                                            model=lr,
                                            label_translation=label_translation)

    ent_st = get_disagreement_statistics(df=dev_b_original.query("label=='entailment'"),
                                         df_t=dev_b_t.query(
                                             "label=='entailment'"),
                                         model=lr,
                                         label_translation=label_translation)

    ent_stats.append(ent_st)
    neutral_stats.append(neutral_st)
    contra_stats.append(contra_st)
    stats.append(st)
    del lr, train_b, dev_b


result = pd.concat(stats)
result.index = ids
result.index.name = "pcts"

ent_result = pd.concat(ent_stats)
ent_result.index = ids
ent_result.index.name = "pcts"

contra_result = pd.concat(contra_stats)
contra_result.index = ids
contra_result.index.name = "pcts"

neutral_result = pd.concat(neutral_stats)
neutral_result.index = ids
neutral_result.index.name = "pcts"


result.to_csv(result_path)
ent_result.to_csv(ent_result_path)
neutral_result.to_csv(neutral_result_path)
contra_result.to_csv(contra_result_path)
