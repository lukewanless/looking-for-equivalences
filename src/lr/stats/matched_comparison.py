import pandas as pd
import numpy as np


def get_results(df, model, label_translation):
    """
    get prediction results from a model 'model'
    and a dataframe 'df'

    :param df: data to predict
    :type df: pd.DataFrame
    :param model: classification model
    :type model: any model from the module 'models'
    :return: results
    :rtype: pd.DataFrame
    """

    pred = model.predict(df)
    label = label_translation(df)
    dict_ = {"label": label,
             "prediction": pred}
    results = pd.DataFrame(dict_)
    results.loc[:, "indicator"] = results.label == results.prediction
    results.loc[:, "indicator"] = results.indicator.apply(lambda x: int(x))
    return results


def get_matched_results(df,
                        df_t,
                        model,
                        label_translation):
    """
    get matched results from a model 'model'
    and dataframes 'df' and 'df_transformed'

    :param df: data to predict
    :type df: pd.DataFrame
    :param df_transformed: data to predict (transformed version)
    :type df_transformed: pd.DataFrame
    :param model: classification model
    :type model: any model from the module 'models'
    :return: matched results
    :rtype: pd.DataFrame
    """
    results = get_results(df, model, label_translation)
    results_t = get_results(df_t, model, label_translation)
    dict_ = {"label": results.label.values,
             "A": results.indicator.values,
             "B": results_t.indicator.values}
    m_results = pd.DataFrame(dict_)
    m_results.loc[:, "acc_minus_acct"] = m_results.A - m_results.B
    return m_results


def a_true_b_true(a, b):
    return int(a == 1 and b == 1)


def a_true_b_false(a, b):
    return int(a == 1 and b == 0)


def a_false_b_true(a, b):
    return int(a == 0 and b == 1)


def a_false_b_false(a, b):
    return int(a == 0 and b == 0)


def get_matched_disagreement(df, df_t, model, label_translation):
    """
    get disagreement iformation on matched samples.

    variable meaning:

    m = df.shape[0]

    A = I(model(P,H) == Y)
    B = I(model(P_t,H_t) == Y_t)

    ABmean = mean(A) - mean(B)
    s = std(A - B)
    sqrt_m = np.sqrt(m)
    se = s / sqrt_m

    t1 = ABmean / se

    C = sum(I(A=1, B=1))
    D = sum(I(A=1, B=0))
    E = sum(I(A=0, B=1))
    F = sum(I(A=0, B=0))

    D + E = total number of disagreement between A and B

    t2 = (D - E)^2 / (D + E)

    pos_trans_rate = E / (D + E)
    neg_trans_rate = D / (D + E)


    :param df: data to predict
    :type df: pd.DataFrame
    :param df_t: data to predict (transformed version)
    :type df_t: pd.DataFrame
    :param model: classification model
    :type model: any model from the module 'models'
    :return: matched results
    :rtype: pd.DataFrame
    """

    results = get_matched_results(df, df_t, model, label_translation)
    n = results.shape[0]
    mean = results.acc_minus_acct.mean()
    std = results.acc_minus_acct.std()
    S = std / np.sqrt(n)
    t1 = mean / S

    c = results.apply(lambda x: a_true_b_true(x.A, x.B), axis=1).sum()
    d = results.apply(lambda x: a_true_b_false(x.A, x.B), axis=1).sum()
    e = results.apply(lambda x: a_false_b_true(x.A, x.B), axis=1).sum()
    f = results.apply(lambda x: a_false_b_false(x.A, x.B), axis=1).sum()
    all_dis = d + e

    t2 = ((d - e) ** 2) / all_dis

    pos_trans_rate = e / all_dis
    neg_trans_rate = d / all_dis

    statistics = [[t1, c, d, e, f, t2, pos_trans_rate, neg_trans_rate]]
    columns = ["t1",
               "C",
               "D",
               "E",
               "F",
               "t2",
               "pos_trans_rate",
               "neg_trans_rate"]
    return pd.DataFrame(statistics, columns=columns)


def get_disagreement_statistics(df,
                                df_t,
                                model,
                                label_translation):
    """
    get disagreement statistics.

    variable meaning:

    A = I(model(P,H) == Y)
    B = I(model(P_t,H_t) == Y_t)
    C = sum(I(A=1, B=1))
    D = sum(I(A=1, B=0))
    E = sum(I(A=0, B=1))
    F = sum(I(A=0, B=0))

    D + E = total number of disagreement between A and B

    t1 = (D - E)^2 / (D + E)
    t2 = (D - E) / (D + E)

    :param df: data to predict
    :type df: pd.DataFrame
    :param df_transformed: data to predict (transformed version)
    :type df_transformed: pd.DataFrame
    :param model: classification model
    :type model: any model from the module 'models'
    :return: matched results
    :rtype: pd.DataFrame
    """

    results = get_results(df, model, label_translation)
    acc = pd.Series(results.indicator.mean())
    acc.name = "acc"
    results_t = get_results(df_t, model, label_translation)
    acc_t = pd.Series(results_t.indicator.mean())
    acc_t.name = "acct"

    mr = get_matched_results(df=df,
                             df_t=df_t,
                             model=model,
                             label_translation=label_translation)
    amb = pd.Series(mr["acc_minus_acct"].mean())
    amb.name = "acc_minus_acct"

    dis = get_matched_disagreement(df=df,
                                   df_t=df_t,
                                   model=model,
                                   label_translation=label_translation)
    comb = [acc, acc_t, amb, dis]

    disagreement_stats = pd.concat(comb, axis=1).reset_index(drop=True)
    return disagreement_stats
