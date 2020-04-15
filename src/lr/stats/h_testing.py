import pandas as pd
import numpy as np
from time import time
from scipy.stats import mode


class DGP():
    """
    data generation process
    """

    def __init__(self,
                 data,
                 transformation,
                 rho):

        self.data = data
        self.transformation = transformation
        self.rho = rho

    def sample(self,
               random_state=None):
        """
        get rho*100% transformed sample from
        data
        """
        df = self.data.copy()
        df_t = self.transformation(df.sample(frac=self.rho,
                                             replace=False,
                                             random_state=random_state))
        safe_ids = [i for i in df.index if i not in df_t.index]
        df_safe = df.iloc[safe_ids]
        return pd.concat([df_t, df_safe]).sort_index()

    def sample_transform(self,
                         random_state=None):
        """
        get rho*100% transformed sample from
        data (for transformer models)
        """
        df = self.data.copy()
        sample = df.sample(frac=self.rho,
                           replace=False,
                           random_state=random_state)
        df_t = self.transformation(sample)
        df_t.loc[:, "o_index"] = sample.o_index
        safe_ids = [i for i in df.index if i not in df_t.o_index]
        df_safe = df.iloc[safe_ids]
        return pd.concat([df_t, df_safe]).sort_index()


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
    return m_results


def get_matched_results_transformers(results,
                                     results_t):
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
    dict_ = {"label": results.label.values,
             "A": results.indicator.values,
             "B": results_t.indicator.values}
    m_results = pd.DataFrame(dict_)
    return m_results


def get_paired_t_statistic(results):
    """
    return t-statisic from paired test:

    np.sqrt(n)*(np.mean(A) - np.mean(B)) / np.std(A - B)
    """

    diff = results.A - results.B
    n = diff.shape[0]
    S = diff.std(ddof=0)
    t = (diff.mean() * np.sqrt(n)) / S
    return t, diff.mean(), n, S


def get_cochran_statistic(results):
    """
    return cochran statisic from paired test:
    """
    crosstab = pd.crosstab(results.A, results.B).values
    error2hit = crosstab[0, 1]
    hit2error = crosstab[1, 0]
    total_error = error2hit + hit2error
    c_stats = ((error2hit - hit2error)**2) / total_error
    return c_stats


def invert_A_B(df):
    """
    invert A and B results
    """
    new_df = df.copy()
    old_A = df.A.values
    old_B = df.B.values
    new_df.loc[:, "A"] = old_B
    new_df.loc[:, "B"] = old_A
    return new_df


def get_boot_sample_under_H0(results, random_state=None):
    """
    generate bootstrap sample under H0: A and B are the same.
    """
    boot_sample = results.sample(
        frac=1,
        replace=True,
        random_state=random_state).reset_index(
        drop=True)
    n = boot_sample.shape[0]
    n_2 = int(n / 2)
    boot_sample_invert = invert_A_B(boot_sample.head(n_2))
    ids = [i for i in boot_sample.index if i not in boot_sample_invert.index]
    boot_H0 = pd.concat([boot_sample_invert,
                         boot_sample.loc[ids]]).reset_index(drop=True)
    return boot_H0


def get_boots_series_under_H0(matched_results, stats_function,
                              number_of_simulations, random_state):

    np.random.seed(random_state)
    t_boots = []

    for _ in range(number_of_simulations):
        boot_sample = get_boot_sample_under_H0(matched_results)
        t = stats_function(boot_sample)
        t_boots.append(t)

    return pd.Series(t_boots)


def get_boot_paired_t_p_value(ts, t_obs):
    """
    ts is a pd.Series
    t_obs is the observable value
     """
    def lower_tail_f(x): return (ts.sort_values() <= x).astype(int).mean()
    def upper_tail_f(x): return (ts.sort_values() > x).astype(int).mean()
    def equal_tail_boot_p_value(x): return 2 * \
        np.min([lower_tail_f(x), upper_tail_f(x)])
    return equal_tail_boot_p_value(t_obs)


def get_boot_cochran_p_value(ts, t_obs):
    def upper_tail_f(x): return (ts.sort_values() > x).astype(int).mean()
    return upper_tail_f(t_obs)


def get_ecdf(series_):
    return lambda x: (series_.sort_values() < x).astype(int).mean()

def cdf_u_0_1(x):
    assert 0 <= x <=1
    return x


def get_ks_stats_from_p_values_compared_to_uniform_dist(p_values, size=100):

    x = np.linspace(0, 1, size)
    ecdf = np.vectorize(get_ecdf(p_values))
    ecdf = np.vectorize(ecdf)
    cdf_u_0_1_v = np.vectorize(cdf_u_0_1)
    y1 = cdf_u_0_1_v(x)
    y2 = ecdf(x)
    diff = np.max(np.abs(y1 - y2))
    return diff