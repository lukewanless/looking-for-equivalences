import re
import logging
import torch
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import random
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.utils import DataProcessor
from torch.utils.data import TensorDataset


logging.basicConfig(filename='example.log', level=logging.INFO)
spaces = re.compile(' +')


def merge_lists(lists):
    base = []
    for l in lists:
        base.extend(l)
    return base


def parallelize_df2list(df, func, n_cores):
    """
    general fucntion to parallelize a function applied to
    a df
    """
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    result = merge_lists(pool.map(func, df_split))
    pool.close()
    pool.join()
    return result


def remove_first_space(x):
    """
    remove_first_space from word x

    :param x: word
    :type x: str
    :return: word withou space in front
    :rtype: str
    """
    try:
        if x[0] == " ":
            return x[1:]
        else:
            return x
    except IndexError:
        return x


def simple_pre_process_text(data, text_column):
    """
    preprocess all input text from dataframe by
    lowering, removing non words, removing
    space in the first position and
    removing double spaces


    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    :param text_column: colum text_column
    :type text_column: str
    """
    s = data.loc[:, text_column].copy()
    s = s.apply(lambda x: x.lower())
    s = s.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))  # noqa
    s = s.apply(remove_first_space)  # noqa remove space in the first position
    s = s.apply((lambda x: spaces.sub(" ", x)))  # noqa remove double spaces
    return s


def pre_process_nli_df(data):
    """
    Apply preprocess on the input text from a NLI dataframe

    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    """
    new_p = simple_pre_process_text(data, text_column="premise")
    new_h = simple_pre_process_text(data, text_column="hypothesis")
    label = data.label
    o_index = data.o_index
    dict_ = {"premise": new_p, "hypothesis": new_h,
             "label": label, "o_index": o_index}
    return pd.DataFrame(dict_)


def filter_df_by_label(df, drop_label='-'):
    """
    drop observations with label 'drop_label'
    """
    return df.loc[df.label != drop_label]


class NLIProcessor(DataProcessor):
    """Processor for the any nli dataf frame in csv
       (columns = premise | hypothesis | label)"""

    def __init__(self, hyperparams):
        super().__init__()
        self.tokenizer = hyperparams["tokenizer"]
        self.max_length = hyperparams["max_seq_length"]
        self.pad_on_left = hyperparams["pad_on_left"]
        self.pad_token = hyperparams["pad_token"]
        self.pad_token_segment_id = hyperparams["pad_token_segment_id"]
        self.mask_padding_with_zero = hyperparams["mask_padding_with_zero"]
        self.base_path = hyperparams["base_path"]

    def df2examples(self, df, set_type):
        df = filter_df_by_label(df.dropna()).reset_index(drop=True)
        df = pre_process_nli_df(df)
        examples = self._create_examples(df, set_type)
        return examples

    def get_train_examples(self, df):
        return self.df2examples(df, "train")

    def get_dev_examples(self, df):
        return self.df2examples(df, "dev")

    def df2examples_parallel_train(self, df, n_cores):
        df_new = df.copy()
        df_new.loc[:, "o_index"] = df.index
        result = parallelize_df2list(df_new, self.get_train_examples, n_cores)
        del df_new
        return result

    def df2examples_parallel_dev(self, df, n_cores):
        df_new = df.copy()
        df_new.loc[:, "o_index"] = df.index
        result = parallelize_df2list(df_new, self.get_dev_examples, n_cores)
        del df_new
        return result

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def get_label_map(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        return label_map

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        n = df.shape[0]
        for i in range(n):
            example = df.loc[i]
            name = example.o_index
            guid = "{}-{}".format(set_type, name)
            input_example = InputExample(guid=guid,
                                         text_a=example.premise,
                                         text_b=example.hypothesis,
                                         label=example.label)

            examples.append(input_example)
        return examples

    def _convert_examples_to_features(self, examples):

        max_length = self.max_length
        pad_token = self.pad_token
        pad_token_segment_id = self.pad_token_segment_id
        mask_padding_with_zero = self.mask_padding_with_zero

        label_map = self.get_label_map()
        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            inputs = self.tokenizer.encode_plus(example.text_a,
                                                example.text_b,
                                                add_special_tokens=True,
                                                max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [
                1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if self.pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1]
                                  * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] *
                                  padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + \
                    ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
                len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
                len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
                len(token_type_ids), max_length)
            label = label_map[example.label]
            features.append(InputFeatures(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          label=label))
        return features

    def examples2features_parallel(self, examples, n_cores):
        result = parallelize_df2list(examples,
                                     self._convert_examples_to_features,
                                     n_cores)
        return result

    def df2features(self, df, n_cores, mode):
        path = self.base_path + "{}_{}".format(mode, self.max_length)
        logging.info("Saving features in file: %s", path)
        if mode.find("train") > -1:
            examples = self.df2examples_parallel_train(df, n_cores)
        else:
            examples = self.df2examples_parallel_dev(df, n_cores)
        features = self.examples2features_parallel(examples, n_cores)
        torch.save(features, path)
        return path


def features2dataset(cached_features_file):
    assert os.path.exists(cached_features_file)
    logging.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels)
    return dataset
