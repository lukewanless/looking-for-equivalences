import pandas as pd
import numpy as np


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

    def get_sample(self):
        """
        get rho*100% transformed sample from
        data
        """
        df = self.data.copy()
        df_t = self.transformation(df.sample(frac=self.rho, replace=False))
        safe_ids = [i for i in df.index if i not in df_t.index]
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


def get_paired_t_statistic(results):
    diff = results.A - results.B
    n = diff.shape[0]
    S = diff.std(ddof=0)
    t = (diff.mean() * np.sqrt(n)) / S
    return t


def invert_A_B(df):
    new_df = df.copy()
    old_A = df.A.values
    old_B = df.B.values
    new_df.loc[:, "A"] = old_B
    new_df.loc[:, "B"] = old_A
    return new_df


def get_boot_sample_under_H0(results):
    boot_sample = results.sample(frac=1, replace=True).reset_index(drop=True)
    boot_sample_invert = invert_A_B(boot_sample.sample(frac=0.5))
    ids = [i for i in boot_sample.index if i not in boot_sample_invert.index]
    boot_H0 = pd.concat([boot_sample_invert,
                         boot_sample.loc[ids]]).reset_index(drop=True)
    return boot_H0


def get_boot_p_value(ts, t_obs):
    """
    ts is a pd.Series
    t_obs is the observable value
     """
    def lower_tail_f(x): return (ts.sort_values() <= x).astype(int).mean()
    def upper_tail_f(x): return (ts.sort_values() > x).astype(int).mean()
    def equal_tail_boot_p_value(x): return 2 * \
        np.min([lower_tail_f(x), upper_tail_f(x)])
    return equal_tail_boot_p_value(t_obs)


def LIMts_test(train, dev, transformation, rho,
               Model, hyperparams, M, E, S):
    dgp = DGP(data=train, transformation=transformation, rho=rho)
    dev_t = transformation(dev)
    index_pair = []
    all_t_obs = []
    all_p_values = []
    all_t_boots = []
    t_columns = ["boot_t_{}".format(i + 1) for i in range(S)]
    for m in range(M):
        train_t = dgp.get_sample()
        for e in range(E):
            index_pair.append((m + 1, e + 1))
            model = Model(hyperparams)
            model.fit(train_t)
            results = get_matched_results(
                dev, dev_t, model, model.label_translation)
            t_obs = get_paired_t_statistic(results)
            all_t_obs.append(t_obs)
            t_boots = []
            for _ in range(S):
                boot_sample = get_boot_sample_under_H0(results)
                t = get_paired_t_statistic(boot_sample)
                t_boots.append(t)
            t_boots = pd.Series(t_boots)
            p_value = get_boot_p_value(t_boots, t_obs)
            all_p_values.append(p_value)
            t_boots_t = t_boots.to_frame().transpose()
            t_boots_t.columns = t_columns
            all_t_boots.append(t_boots_t)

    dict_ = {"experiment_index": index_pair,
             "observable_t_stats": all_t_obs,
             "p_value": all_p_values}

    test_results = pd.DataFrame(dict_)
    t_boots_df = pd.concat(all_t_boots).reset_index(drop=True)
    combined_information = pd.merge(test_results,
                                    t_boots_df,
                                    right_index=True,
                                    left_index=True)
    return combined_information
