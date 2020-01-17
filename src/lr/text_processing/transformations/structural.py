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


def label_internalization(p, h, l):
    """
    transformation that adds the label to the pair (p,h)


    :param p: premise
    :type p: str
    :param h: hypothesis
    :type h: str
    :param l: label
    :type l: str
    :return: new observation (p_new,h_new,l_new)
    :rtype: (str,str,str)
    """
    new_p = p
    new_h = h + " , {} ,".format(l)
    return p, new_h, l