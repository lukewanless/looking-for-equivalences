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
    return t


def get_paired_t_statistic_full(results):
    """
    return t-statisic from paired test:

    np.sqrt(n)*(np.mean(A) - np.mean(B)) / np.std(A - B)
    """

    diff = results.A - results.B
    n = diff.shape[0]
    S = diff.std(ddof=0)
    t = (diff.mean() * np.sqrt(n)) / S
    return t, diff.mean(), n, S


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
               Model,
               hyperparams):

    path_results_base = hyperparams["output_dir"] + "/results_"
    path_best_params = hyperparams["output_dir"] + "/best_params"
    random_state_list = hyperparams["random_state_list"]
    dgp_seed_list = hyperparams["dgp_seed_list"]
    data_set_name = hyperparams["data_set_name"]
    transformation_name = hyperparams["transformation_name"]
    rho = hyperparams["rho"]
    model_name = hyperparams["model_name_or_path"]
    M = hyperparams["number_of_samples"]
    E = hyperparams["number_of_models"]
    S = hyperparams["number_of_simulations"]
    verbose = hyperparams["verbose"]

    # seeds for dgp
    if dgp_seed_list is None:
        dgp_seed_list = [None] * M

    if random_state_list is None:
        random_state_list = [None] * M

    # intial setting
    dgp = DGP(data=train,
              transformation=train_transformation,
              rho=rho)
    t_columns = ["boot_t_{}".format(i + 1) for i in range(S)]
    dev_t = dev_transformation(dev)

    all_t_obs = []
    all_acc_diffs = []
    all_test_sizes = []
    all_standart_errors = []
    majority_accs = []
    majority_accs_t = []
    all_p_values = []
    all_t_boots = []
    models_train_acc_mean = []
    models_train_acc_std = []
    htest_times = []
    train_times = []

    # generate modified training sample
    for i, m in enumerate(range(M)):
        train_t = dgp.sample(random_state=dgp_seed_list[i])
        all_models = []
        init_test = time()
        init_train = time()
        # setting seed
        np.random.seed(random_state_list[i])

        # train model
        model = Model(hyperparams)
        model.fit(train_t)
        train_time = time() - init_train
        train_times.append(train_time)

        # Save best params Define the majority model
        best_assigment = model.model.best_params_
        times = model.model.cv_results_['mean_fit_time']
        mean_time = np.mean(times)
        n_trains = len(times)
        with open(path_best_params + "_{}.txt".format(m), "w") as file:
            for key in best_assigment:
                file.write("{} = {}\n".format(key, best_assigment[key]))
            file.write("\nbest_acc = {:.1%}".format(model.model.best_score_))
            file.write("\ntime = {:.1f} s".format(mean_time))
            file.write("\nnumber of search trials = {}".format(n_trains))
        
        # Get observed accs and t stats
        results = get_matched_results(
            dev, dev_t, model, model.label_translation)

        path_results = path_results_base + "{}.csv".format(m)
        results.to_csv(path_results)

        majority_accs.append(results.A.mean())
        majority_accs_t.append(results.B.mean())
        t_obs, acc_diff, test_size, standart_error = get_paired_t_statistic_full(
            results)

        all_t_obs.append(t_obs)
        all_acc_diffs.append(acc_diff)
        all_test_sizes.append(test_size)
        all_standart_errors.append(standart_error)

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
            print(
                "m = {} | time: {:.1f} minutes".format(
                    m + 1, test_time / 60))

    dict_ = {"data": [data_set_name] * M,
             "model": [model_name] * M,
             "transformation": [transformation_name] * M,
             "rho": [rho] * M,
             "dgp_seed": dgp_seed_list,
             "random_state": random_state_list,
             "number_of_simulations": [S] * M,
             "validation_accuracy": majority_accs,
             "transformed_validation_accuracy": majority_accs_t,
             "accuracy_difference": all_acc_diffs,
             "test_size": all_test_sizes,
             "standart_error": all_standart_errors,
             "observable_t_stats": all_t_obs,
             "p_value": all_p_values,
             "training_time": train_times,
             "test_time": htest_times}

    test_results = pd.DataFrame(dict_)
    t_boots_df = pd.concat(all_t_boots).reset_index(drop=True)
    combined_information = pd.merge(test_results,
                                    t_boots_df,
                                    right_index=True,
                                    left_index=True)
    return combined_information


def h_test_transformer(df_train,
                       df_dev,
                       df_dev_t,
                       ModelWrapper,
                       hyperparams):

    random_state = hyperparams["random_state"]
    dgp_seed = hyperparams["dgp_seed"]
    data_set_name = hyperparams["data_set_name"]
    transformation_name = hyperparams["transformation_name"]
    rho = hyperparams["rho"]
    path_results = hyperparams["output_dir"] + "/results.csv"
    model_name = hyperparams["model_name_or_path"]
    S = hyperparams["number_of_simulations"]

    init = time()
    transformer = ModelWrapper(hyperparams)

    global_step, tr_loss, train_time = transformer.fit(df_train)

    dev_results = transformer.get_results(df_dev, mode="test")
    dev_t_results = transformer.get_results(df_dev_t, mode="test_t")

    m_results = get_matched_results_transformers(dev_results, dev_t_results)
    m_results.to_csv(path_results)
    t_obs, acc_diff, test_size, standart_error = get_paired_t_statistic_full(
        m_results)

    if random_state is not None:
        np.random.seed(random_state)

    # Generate S bootstrap replications
    t_boots = []
    for _ in range(S):
        boot_sample = get_boot_sample_under_H0(m_results)
        t = get_paired_t_statistic(boot_sample)
        t_boots.append(t)

    # Get bootstrap p-value
    t_boots = pd.Series(t_boots)
    p_value = get_boot_p_value(t_boots, t_obs)

    htest_time = time() - init

    # Aggregate all results
    dict_ = {"data": [data_set_name],
             "model": [model_name],
             "transformation": [transformation_name],
             "rho": [rho],
             "dgp_seed": [dgp_seed],
             "random_state": [random_state],
             "number_of_simulations": [S],
             "validation_accuracy": [m_results.A.mean()],
             "transformed_validation_accuracy": [m_results.B.mean()],
             "accuracy_difference": [acc_diff],
             "test_size": [test_size],
             "standart_error": [standart_error],
             "observable_t_stats": [t_obs],
             "p_value": [p_value],
             "training_time": [train_time / 3600],
             "test_time": [htest_time / 3600]}

    test_results = pd.DataFrame(dict_)

    t_columns = ["boot_t_{}".format(i + 1) for i in range(S)]
    t_boots_df = t_boots.to_frame().transpose()
    t_boots_df.columns = t_columns

    combined_information = pd.merge(test_results,
                                    t_boots_df,
                                    right_index=True,
                                    left_index=True)
    return combined_information


def h_test_transformer_trained_model(df_dev,
                                     df_dev_t,
                                     transformer,
                                     hyperparams):

    random_state = hyperparams["random_state"]
    dgp_seed = hyperparams["dgp_seed"]
    data_set_name = hyperparams["data_set_name"]
    transformation_name = hyperparams["transformation_name"]
    rho = hyperparams["rho"]
    path_results = hyperparams["output_dir"] + "/results.csv"
    model_name = hyperparams["model_name_or_path"]
    S = hyperparams["number_of_simulations"]

    init = time()

    train_time = np.nan

    dev_results = transformer.get_results(df_dev, mode="test")
    dev_t_results = transformer.get_results(df_dev_t, mode="test_t")

    m_results = get_matched_results_transformers(dev_results, dev_t_results)
    m_results.to_csv(path_results)
    t_obs, acc_diff, test_size, standart_error = get_paired_t_statistic_full(
        m_results)

    if random_state is not None:
        np.random.seed(random_state)

    # Generate S bootstrap replications
    t_boots = []
    for _ in range(S):
        boot_sample = get_boot_sample_under_H0(m_results)
        t = get_paired_t_statistic(boot_sample)
        t_boots.append(t)

    # Get bootstrap p-value
    t_boots = pd.Series(t_boots)
    p_value = get_boot_p_value(t_boots, t_obs)

    htest_time = time() - init

    # Aggregate all results
    dict_ = {"data": [data_set_name],
             "model": [model_name],
             "transformation": [transformation_name],
             "rho": [rho],
             "dgp_seed": [dgp_seed],
             "random_state": [random_state],
             "number_of_simulations": [S],
             "validation_accuracy": [m_results.A.mean()],
             "transformed_validation_accuracy": [m_results.B.mean()],
             "accuracy_difference": [acc_diff],
             "test_size": [test_size],
             "standart_error": [standart_error],
             "observable_t_stats": [t_obs],
             "p_value": [p_value],
             "training_time": [train_time / 3600],
             "test_time": [htest_time / 3600]}

    test_results = pd.DataFrame(dict_)

    t_columns = ["boot_t_{}".format(i + 1) for i in range(S)]
    t_boots_df = t_boots.to_frame().transpose()
    t_boots_df.columns = t_columns

    combined_information = pd.merge(test_results,
                                    t_boots_df,
                                    right_index=True,
                                    left_index=True)
    return combined_information


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


def get_p_value_cochran(ts, t_obs):
    def upper_tail_f(x): return (ts.sort_values() > x).astype(int).mean()
    return upper_tail_f(t_obs)


def get_c_stats_from_result(meta_results_description_row, results):

    random_state = meta_results_description_row.random_state[0]
    S = meta_results_description_row.number_of_simulations[0]
    np.random.seed(random_state)
    c_columns = ["boot_c_{}".format(i + 1) for i in range(S)]
    # Generate S bootstrap replications

    c_boots = []
    for _ in range(S):
        boot_sample = get_boot_sample_under_H0(results)
        c = get_cochran_statistic(boot_sample)
        c_boots.append(c)
    c_boots = pd.Series(c_boots, index=c_columns)

    c_obs = get_cochran_statistic(results)
    p_value = get_p_value_cochran(c_boots, c_obs)
    return c_obs, p_value, c_boots


def update_results_with_cochran_test(meta_results_description_row, results):

    c_obs, p_value, c_boots = get_c_stats_from_result(
        meta_results_description_row, results)

    c_boots = c_boots.to_frame().transpose()
    new = pd.DataFrame({"cochran_statistic": c_obs,
                        "cochran_p_value": p_value}, index=[0])
    new = pd.merge(new, c_boots, left_index=True, right_index=True)
    new_meta = pd.merge(
        meta_results_description_row,
        new,
        left_index=True,
        right_index=True)
    return new_meta
