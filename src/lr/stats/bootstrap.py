def get_bootstrap_replications(df):
    """
    let n = df.shape[0], this function
    draws n observation with replacement from df

    :param df: data
    :type df: pd.DataFrame
    :return: simulated data
    :rtype: pd.DataFrame
    """
    return df.sample(replace=True, frac=1)


def apply_transformation_under_H0(df, df_transformation_f):
    """
    under H0 noise data and original data should be classiefied
    as the same. Hence this functions creates a original and
    a transformed data where the transformed observations
    appear on both df and df_t


    :param df: data
    :type df: pd.DataFrame
    :param transformation:  transformation noise
    :type transformation: function
    :return: simulated data
    :rtype: pd.DataFrame
    """
    df = df.reset_index(drop=True)
    df_t = df_transformation_f(df)
    id_ = list(df.sample(replace=False, frac=0.5).index)
    original_lines = df.loc[id_, :].copy()
    transformed_lines =  df_t.loc[id_, :].copy()
    df.loc[id_, :] = transformed_lines
    df_t.loc[id_, :] = original_lines
    return df, df_t