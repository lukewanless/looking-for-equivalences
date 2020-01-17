import pandas as pd


def apply_syn(df, text_column, syn):
    def f(x): return syn[x] if x in syn else x
    def f_str(s): return " ".join(map(f, s.split()))
    return df.loc[:, text_column].apply(f_str)


def transform_syn_p(df, syn):
    df.loc[:, "premise"] = apply_syn(df, "premise", syn)


def transform_syn_h(df, syn):
    df.loc[:, "hypothesis"] = apply_syn(df, "hypothesis", syn)


def transform_syn_p_h(df, syn):
    transform_syn_p(df, syn)
    transform_syn_h(df, syn)


def get_transformed_part_by_syn(df, syn):
    diff_p = df.premise != apply_syn(df, "premise", syn)
    diff_h = df.hypothesis != apply_syn(df, "hypothesis", syn)
    select = diff_p | diff_h
    return df.loc[select]


def syn2tranformation(syn):
    return lambda x: transform_syn_p_h(x, syn)


def get_augmented_data(df, transformation, frac):
    df_tranformed = df.sample(frac=frac, replace=False)
    safe_ids = [i for i in df.index if i not in df_tranformed.index]
    df_safe = df.iloc[safe_ids]
    transformation(df_tranformed)
    return pd.concat([df_tranformed, df_safe]).sort_index()
