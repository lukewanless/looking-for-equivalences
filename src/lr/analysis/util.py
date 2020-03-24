import pandas as pd
from IPython.display import display, HTML

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