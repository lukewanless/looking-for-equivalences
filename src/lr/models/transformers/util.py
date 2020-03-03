from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.utils import DataProcessor
from torch.utils.data import TensorDataset
import torch
import re
import logging
import os
import pandas as pd
import numpy as np

logging.basicConfig(filename='example.log', level=logging.INFO)
spaces = re.compile(' +')


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


def simple_pre_process_text_df(data, text_column):
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

    data.loc[:, text_column] = data.loc[:,
                                        text_column].apply(lambda x: x.lower())
    data.loc[:, text_column] = data.loc[:, text_column].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))  # noqa
    data.loc[:, text_column] = data.loc[:, text_column].apply(remove_first_space)  # noqa remove space in the first position
    data.loc[:, text_column] = data.loc[:, text_column].apply((lambda x: spaces.sub(" ", x)))  # noqa remove double spaces


def pre_process_nli_df(data):
    """
    Apply preprocess on the input text from a NLI dataframe

    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    """
    simple_pre_process_text_df(data, text_column="premise")
    simple_pre_process_text_df(data, text_column="hypothesis")


def filter_df_by_label(df, drop_label='-'):
    """
    drop observations with label 'drop_label'
    """
    return df.loc[df.label != drop_label]


class NLIProcessor(DataProcessor):
    """Processor for the any nli dataf frame in csv
       (columns = premise | hypothesis | label)"""

    def read_and_clean_csv(self, path):
        df = pd.read_csv(path)
        df = filter_df_by_label(df.dropna()).reset_index(drop=True)
        pre_process_nli_df(df)
        return df

    def get_train_examples(self, path):
        logging.info("creating train examples for {}".format(path))
        return self._create_examples(self.read_and_clean_csv(path), "train")

    def get_dev_examples(self, path):
        logging.info("creating dev examples for {}".format(path))
        return self._create_examples(self.read_and_clean_csv(path), "dev")

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
            guid = "{}-{}".format(set_type, example.name)
            input_example = InputExample(guid=guid,
                                         text_a=example.premise,
                                         text_b=example.hypothesis,
                                         label=example.label)

            examples.append(input_example)
        return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 label_map,
                                 max_length=512,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 100 == 0:
            logging.info("converting example %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(example.text_a,
                                       example.text_b,
                                       add_special_tokens=True,
                                       max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
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


def load_and_cache_examples(hyperparams, tokenizer, evaluate=False):
    overwrite_cache = hyperparams["overwrite_cache"]
    max_seq_length = hyperparams["max_seq_length"]
    local_rank = hyperparams["local_rank"]
    cached_path = hyperparams["cached_path"]
    train_path = hyperparams["train_path"]
    dev_path = hyperparams["dev_path"]

    processor = NLIProcessor()
    label_map = processor.get_label_map()

    if local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    mode = "dev" if evaluate else "train"
    cached_features_file = cached_path + \
        "cached_{}_{}".format(mode, max_seq_length)

    if os.path.exists(cached_features_file) and not overwrite_cache:
        logging.info(
            "Loading features from cached file %s",
            cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logging.info(
            "Creating features from dataset file at %s",
            cached_features_file)
        if evaluate:
            examples = processor.get_dev_examples(dev_path)
        else:
            examples = processor.get_train_examples(train_path)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_map=label_map,
                                                max_length=max_seq_length)
        if local_rank in [-1, 0]:
            logging.info(
                "Saving features into cached file %s",
                cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if local_rank == 0 and not evaluate:
        torch.distributed.barrier()

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
