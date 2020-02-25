import pandas as pd
import numpy as np
from time import time
from scipy.stats import mode


class Majority():
    """
    Classifiers majority vote
    """

    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.label_translation = classifiers[0].label_translation

    def predict(self, df):
        predictions = [c.predict(df) for c in self.classifiers]
        predictions = np.stack(predictions).T
        predictions = np.array([mode(p).mode[0] for p in predictions])
        return predictions

    def get_acc(self, df):
        x = self.predict(df)
        y = self.label_translation(df)
        return np.mean(x == y)


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
    """
    return t-statisic from paired test:

    np.sqrt(n)*(np.mean(A) - np.mean(B)) / np.std(A - B)
    """

    diff = results.A - results.B
    n = diff.shape[0]
    S = diff.std(ddof=0)
    t = (diff.mean() * np.sqrt(n)) / S
    return t


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


def LIMts_test(train,
               dev,
               train_transformation,
               dev_transformation,
               rho,
               Model,
               hyperparams,
               M, E, S,
               verbose=False,
               random_state=None):

    # setting seed

    if random_state is not None:
        np.random.seed(random_state)

    # intial setting
    dgp = DGP(data=train,
              transformation=train_transformation,
              rho=rho)
    t_columns = ["boot_t_{}".format(i + 1) for i in range(S)]
    dev_t = dev_transformation(dev)

    all_t_obs = []
    majority_accs = []
    majority_accs_t = []
    all_p_values = []
    all_t_boots = []
    all_Ms = []
    models_train_acc_mean = []
    models_train_acc_std = []
    htest_times = []
    train_times = []
    trasformation_times = []

    # generate modified training sample
    for m in range(M):
        init = time()
        train_t = dgp.sample()
        t_time = time() - init
        trasformation_times.append(t_time)

        all_models = []
        all_Ms.append(m + 1)
        init_test = time()
        init_train = time()

    # train E models on the same data
        for e in range(E):
            model = Model(hyperparams)
            model.fit(train_t)
            all_models.append(model)
        train_time = time() - init_train
        train_times.append(train_time)

        # Define the majority model
        all_models_train_acc = [m.get_score() for m in all_models]
        models_train_acc_mean.append(np.mean(all_models_train_acc))
        models_train_acc_std.append(np.std(all_models_train_acc))

        # Get observed accs and t stats
        m_model = Majority(all_models)
        results = get_matched_results(
            dev, dev_t, m_model, m_model.label_translation)
        majority_accs.append(results.A.mean())
        majority_accs_t.append(results.B.mean())
        t_obs = get_paired_t_statistic(results)
        all_t_obs.append(t_obs)

        # Generate S bootstrap replications
        t_boots = []
        for _ in range(S):
            boot_sample = get_boot_sample_under_H0(results)
            t = get_paired_t_statistic(boot_sample)
            t_boots.append(t)

        # Get bootstrap p-value
        t_boots = pd.Series(t_boots)
        p_value = get_boot_p_value(t_boots, t_obs)
        all_p_values.append(p_value)
        t_boots_t = t_boots.to_frame().transpose()
        t_boots_t.columns = t_columns
        all_t_boots.append(t_boots_t)
        test_time = time() - init_test
        htest_times.append(test_time)
        if verbose:
            print("m = {} | time: {:.2f} sec".format(m + 1, test_time))

    dict_ = {"m": all_Ms,
             "train_accuracy_mean": models_train_acc_mean,
             "train_accuracy_std": models_train_acc_std,
             "validation_accuracy": majority_accs,
             "transformed_validation_accuracy": majority_accs_t,
             "observable_t_stats": all_t_obs,
             "p_value": all_p_values,
             "transformation_time": trasformation_times,
             "training_time": train_times,
             "test_time": htest_times}

    test_results = pd.DataFrame(dict_)
    t_boots_df = pd.concat(all_t_boots).reset_index(drop=True)
    combined_information = pd.merge(test_results,
                                    t_boots_df,
                                    right_index=True,
                                    left_index=True)
    return combined_information
