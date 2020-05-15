import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()


parser.add_argument('folder',
                    type=str,
                    help='data folder')

parser.add_argument('search_random_state',
                    type=int,
                    help='random_state for hyperparams search')

parser.add_argument('batch',
                    type=int,
                    help='test batch')

parser.add_argument('n_cores',
                    type=int,
                    help='n_cores')

args = parser.parse_args()
folder = args.folder
search_random_state = args.search_random_state
b = args.batch
n_cores = args.n_cores


r_path = "results/{}/albert_base/syn_p_h/batch{}".format(folder, b)
rr_path = "raw_results/{}/albert_base/syn_p_h/batch{}".format(folder, b)

if not os.path.exists(r_path):
    os.mkdir(r_path)
else:
    exit()

if not os.path.exists(rr_path):
    os.mkdir(rr_path)
else:
    exit()


all_rhos = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
dgp_seeds = range(1, 50000)
train_seeds = range(1, 50000)
boot_seeds = range(1, 50000)

assert os.path.exists(r_path)
assert len(os.listdir(r_path)) == 0
assert os.path.exists(rr_path)
assert len(os.listdir(rr_path)) == 0

with open("albert_script.sh", "w") as file:

    for rho in all_rhos:
        dgp = np.random.choice(dgp_seeds)
        train = np.random.choice(train_seeds)
        boot = np.random.choice(boot_seeds)

        command = "python3 wordnet_syn_test_albert_base.py {} {:.2f} {} {} {} {} {}\n".format(folder,
                                                                                              rho,
                                                                                              search_random_state,
                                                                                              dgp,
                                                                                              train,
                                                                                              boot,
                                                                                              n_cores)
        file.write(command)
    file.write(
        "mv results/{0}/albert_base/syn_p_h/rho_* results/{0}/albert_base/syn_p_h/batch{1}\n".format(folder, b))
    file.write(
        "mv raw_results/{0}/albert_base/syn_p_h/rho_* raw_results/{0}/albert_base/syn_p_h/batch{1}".format(folder, b))
