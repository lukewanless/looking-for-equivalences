import pandas as pd

def invert(df):
    """
    transformation that returns h, p, l
    for l in {-1. 0}

    """
    df_not_ent = df.query("label!=1").copy()
    df_ent = df.query("label==1").copy()
    new_p = df_not_ent.loc[:, "hypothesis"].values.copy()
    new_h = df_not_ent.loc[:, "premise"].values.copy()
    df_not_ent.loc[:, "premise"] = new_p
    df_not_ent.loc[:, "hypothesis"] = new_h
    return pd.concat([df_ent, df_not_ent]).sort_index()


def entailment_internalization(df):
    """
    new_ p = ''
    new_h = 'p implies that h' (1,0)
    new_h = 'p and h' (-1)
    """
    contra_combine = " and "
    not_contra_combine = " implies that "
    df_not_contra = df.query("label!=-1").copy()
    df_contra = df.query("label==-1").copy()
    
    combs = [contra_combine, not_contra_combine]
    dfs = [df_contra, df_not_contra]

    for comb, df_ in zip(combs, dfs):
        new_p = [""] * df_.shape[0]
        new_h = df_.premise.values + comb + df_.hypothesis.values
        df_.loc[:, "premise"] = new_p
        df_.loc[:, "hypothesis"] = new_h

    return pd.concat(dfs).sort_index()