from multiprocessing import Pool
import numpy as np
import pandas as pd
import copy
import spacy
from nltk.corpus import wordnet, stopwords
from nltk.metrics import edit_distance


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


def wordnet_modifier(sentence,
                     text_processing_pipeline,
                     tag="NOUN",
                     dist_id=0):
    doc = text_processing_pipeline(sentence)
    modified_sentence = copy.copy(sentence)
    for token in doc:
        if token.pos_ == tag and not token.is_stop:
            syn_list = sort_syn_by_dist(token.text)

            if len(syn_list) > 0:
                syn = sort_syn_by_dist(token.text)[dist_id]
                syn = syn.lower()
                modified_sentence = modified_sentence.replace(token.text, syn)

    return modified_sentence


nlp = spacy.load("en_core_web_sm")


def p_h_transformation_noun_minimal_edition(df):
    combined = df.premise + " [SEP] " + df.hypothesis
    combined = combined.map(
        lambda x: wordnet_modifier(
            x, nlp, tag="NOUN", dist_id=0))
    premise = combined.map(lambda x: x.split("[SEP]")[0])
    hypothesis = combined.map(lambda x: x.split("[SEP]")[1])
    df_new = pd.DataFrame({"premise": premise,
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


def path_base_transormation(df, path):
    df_t = pd.read_csv(path)
    ids = df.index
    return df_t.loc[ids]