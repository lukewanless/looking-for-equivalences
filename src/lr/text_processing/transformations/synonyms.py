import pandas as pd


def apply_syn(df, text_column, syn):
    def f(x): return syn[x] if x in syn else x
    def f_str(s): return " ".join(map(f, s.split()))
    return df.loc[:, text_column].apply(f_str)


def transform_syn_p_(df, syn):
    df.loc[:, "premise"] = apply_syn(df, "premise", syn)


def transform_syn_h_(df, syn):
    df.loc[:, "hypothesis"] = apply_syn(df, "hypothesis", syn)


def transform_syn_p(df, syn):
    df_t = df.copy()
    transform_syn_p_(df_t, syn)
    return df_t


toy = {"man": "guy",
       "city": "town",
       "woman": "gal"}

def toy_transformation(df):
    return transform_syn_p(df, toy)
