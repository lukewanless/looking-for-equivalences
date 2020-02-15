def select_str(str_):
    """
    select_str(str_) is a function that
    when applied to x returns true if x contains
    any occurence of str_

    Examples:
    select_str("fox")("The quick brown fox jumps over the lazy dog") = True
    select_str("car")("The quick brown fox jumps over the lazy dog") = False

    :param str_: input string
    :type str_: str
    :return: indicator function for strings
    :rtype: function
    """
    return lambda x: (x.lower().find(str_.lower()) > -
                      1) if isinstance(x, str) else False


def column_cropper(df, str_):
    """
    crop the dataframe 'df' by selecting
    only the columns that contains the substring
    'str_' .

    :param df: data frame
    :type df: pd.DataFrame
    :param str_: input string
    :type str_: str
    :return: cropped dataframe
    :rtype: pd.DataFrame
    """
    return df.loc[:, [c for c in list(df.columns) if select_str(str_)(c)]]