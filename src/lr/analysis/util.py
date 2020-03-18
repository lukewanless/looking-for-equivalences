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

# def show_test(df):
#     t_boots = get_boots(df)
#     t_obs = df.observable_t_stats[0]
#     p_value = df.p_value[0]
#     new_p = get_boot_p_value(t_boots, t_obs)
#     assert new_p == p_value
#     fig, ax = plt.subplots(figsize=(10,5))
#     t_boots.hist(ax=ax, label="bootstrap replications");
#     plt.axvline(x=t_obs, color='r', linestyle='-', label="observed (t={:.1f})".format(t_obs));
#     ax.set_xlabel("t", fontsize=14);
#     ax.set_ylabel("frequency", fontsize=14);
#     ax.set_title("Bootstrap test histogram (p-value = {:.8f})".format(p_value) +"\n", fontsize=16)
#     plt.legend(loc="best");
