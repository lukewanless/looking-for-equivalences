import spacy
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from functools import reduce
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet, stopwords
from nltk.metrics import edit_distance

nlp = spacy.load("en_core_web_sm")


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


def count_word_tag(corpus,
                   text_processing_pipeline,
                   tag):
    word_by_tag = []
    for sentence in corpus:
        doc = text_processing_pipeline(sentence)
        for token in doc:
            if token.pos_ == tag:
                word_by_tag.append(token.text)
    return Counter(word_by_tag)


def count_noun(df):
    return count_word_tag(df,
                          text_processing_pipeline=nlp,
                          tag="NOUN")


def parallel_word_count(df, n_cores, count_func):
    df_split = np.array_split(df, n_cores)
    corpus_split = list(map(get_corpus, df_split))
    pool = Pool(n_cores)
    result = pool.map(count_func, corpus_split)
    result = reduce((lambda x, y: x + y), result)
    pool.close()
    pool.join()
    return result


def parallel_count_noun(df, n_cores):
    return parallel_word_count(df,
                               n_cores,
                               count_func=count_noun)


def get_all_syns(w, choice=0):
    syns = wordnet.synsets(w)
    try:
        syns = syns[choice].lemmas()
        syns = [s.name().replace("_", " ") for s in syns]
        syns = [s for s in syns if s != w]
    except IndexError:
        pass
    return syns


def sort_syn_by_dist(w, choice=0, reverse=False):
    syns = get_all_syns(w, choice)
    syns_d = list(map(lambda x: edit_distance(w, x), syns))
    pairs = list(zip(syns_d, syns))
    pairs.sort(reverse=reverse)
    return [p[1] for p in pairs]


def get_syn_from_freq(word, freq_dict):
    syns = get_all_syns(word)
    if len(syns) > 0:
        syns_present = [s for s in syns if s in freq_dict]
        if len(syns_present) > 0:
            syns_present = [(freq_dict[s], s) for s in syns_present]
            syns_present.sort(reverse=True)
            syn = syns_present[0][1]
        else:
            syns_d = list(map(lambda x: edit_distance(word, x), syns))
            pairs = sorted(zip(syns_d, syns))
            syn = pairs[0][1]
    else:
        syn = word
    return syn.lower()


def get_syn_dict(freq_dict):
    stemmer = PorterStemmer()
    syn_dict = {}
    for word in freq_dict:
        syn = get_syn_from_freq(word, freq_dict)
        syn_stem = stemmer.stem(syn)
        word_stem = stemmer.stem(word)
        if syn_stem != word_stem:
            syn_dict[word] = syn
    return syn_dict


def get_noun_syn_dict(df, n_cores):
    noun_count = parallel_count_noun(df=df, n_cores=n_cores)
    syn_dict = get_syn_dict(noun_count)
    return syn_dict


def transform_using_syn_dict(sentence, syn_dict):
    new_sentence = []
    for w in sentence.split():
        if w in syn_dict:
            new_w = syn_dict[w]
        else:
            new_w = w
        new_sentence.append(new_w)
    return " ".join(new_sentence)


def p_h_transformation_syn_dict(df, syn_dict):
    combined = df.premise + " [SEP] " + df.hypothesis
    combined = combined.map(lambda s: transform_using_syn_dict(s, syn_dict))
    premise = combined.map(lambda x: x.split("[SEP]")[0])
    hypothesis = combined.map(lambda x: x.split("[SEP]")[1])
    df_new = pd.DataFrame({"premise": premise,
                           "hypothesis": hypothesis,
                           "label": df.label})
    return df_new


def p_transformation_syn_dict(df, syn_dict):
    premise = df.premise.map(lambda s: transform_using_syn_dict(s, syn_dict))
    df_new = pd.DataFrame({"premise": premise,
                           "hypothesis": df.hypothesis,
                           "label": df.label})
    return df_new


def h_transformation_syn_dict(df, syn_dict):
    hypothesis = df.hypothesis.map(
        lambda s: transform_using_syn_dict(
            s, syn_dict))
    df_new = pd.DataFrame({"premise": df.premise,
                           "hypothesis": hypothesis,
                           "label": df.label})
    return df_new


def parallelize(df, func, n_cores):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
