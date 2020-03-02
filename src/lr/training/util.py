def filter_df_by_label(df, drop_label='-'):
    """
    drop observations with label 'drop_label'
    """
    return df.loc[df.label != drop_label]


def get_binary_label(df):
    """
    get binary label from nli labels
    """
    nli2binary = {"entailment": 1,
                  "contradiction": 0,
                  "neutral": 0, }
    return df.label.apply(lambda x: nli2binary[x]).values


def get_ternary_label(df):
    """
    get ternary label from nli labels
    """
    nli2ternary = {"entailment": 1,
                   "contradiction": -1,
                   "neutral": 0, }
    return df.label.apply(lambda x: nli2ternary[x]).values



def get_positive_labels(df):
    """
    get ternary label from nli labels
    using only positive numbers
    """
    nli2ternary = {"entailment": 1,
                   "contradiction": 0,
                   "neutral": 2, }
    return df.label.apply(lambda x: nli2ternary[x]).values




def label2binary_label(df):
    """
    get binary label from nli labels
    """
    nli2binary = {"entailment": 1,
                  "contradiction": 0,
                  "neutral": 0}

    df.loc[:, "label"] = df.label.apply(lambda x: nli2binary[x]).values


def label2ternary_label(df):
    """
    get ternary label from nli labels
    """
    nli2ternary = {"entailment": 1,
                   "contradiction": -1,
                   "neutral": 0}

    df.loc[:, "label"] = df.label.apply(lambda x: nli2ternary[x]).values
