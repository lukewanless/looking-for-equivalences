import re
import pandas as pd
from functools import reduce
from collections import Counter

spaces = re.compile(' +')


def get_corpus(df):
    """
    get corpus from NLI dataset

    :param df: NLI df
    :type df: pd.DataFrame
    :return: corpus, list of sentences
    :rtype: [str]
    """
    corpus = df.premise + " " + df.hypothesis
    return list(corpus.values)


def get_vocab_count(df):
    """
    get word frequency from corpus

    :param df: NLI df
    :type df: pd.DataFrame
    :return: word Counter
    :rtype: Counter
    """
    words = reduce((lambda x, y: x + y),
                   map(lambda x: x.split(), get_corpus(df)))
    return Counter(words)


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
