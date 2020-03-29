import pandas as pd
from IPython.display import display, HTML
import numpy as np

basic_columns = ['data',
                 'model',
                 'transformation',
                 'rho',
                 'dgp_seed',
                 'random_state',
                 'number_of_simulations',
                 'validation_accuracy',
                 'transformed_validation_accuracy',
                 'accuracy_difference',
                 'test_size',
                 'standart_error',
                 'observable_t_stats',
                 'p_value',
                 'cochran_statistic',
                 'cochran_p_value',
                 'training_time',
                 'test_time']


def get_not_boot_columns(df):
    return [c for c in df.columns if c.find("boot") == -1]


def get_boot_columns(df):
    return [c for c in df.columns if c.find("boot") > -1]


def get_standart_results(df):
    boot_columns = get_boot_columns(df)
    columns = basic_columns + boot_columns
    return df[columns]


def show_df(df):
    not_boot_columns = get_not_boot_columns(df)
    display(HTML(df[not_boot_columns].to_html()))


def get_boots(df):
    boot_columns = get_boot_columns(df)
    boots = df[boot_columns].transpose().reset_index(drop=True)
    boots = boots[0]
    boots.name = "boots_rho_{}_dgp_seed_{}_rs_{}".format(df.rho[0],
                                                         df.dgp_seed[0],
                                                         df.random_state[0])
    return boots


def get_mismatch(df, name):
    indicator = df.A.map(lambda x: str(x)) + "_" + df.B.map(lambda x: str(x))
    indicator = (indicator == "1_0").astype(np.int)
    indicator.name = name
    return indicator.to_frame()


def join_df_list(df_list):
    """
    Performs the merge_asof in a list of dfs

    :param df_list: list of data frames
    :type df_list: [pd.DataFrame]
    :return: merged dataframe
    :rtype: pd.DataFrame
    """
    size = len(df_list)
    df = pd.merge(df_list[0], df_list[1],
                  left_index=True,
                  right_index=True)
    for i in range(2, size):
        df = pd.merge_asof(df, df_list[i],
                           left_index=True,
                           right_index=True)
    return df
