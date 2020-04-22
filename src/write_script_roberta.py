import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('batch',
                    type=int,
                    help='test batch')

parser.add_argument('n_cores',
                    type=int,
                    help='n_cores')

args = parser.parse_args()

b = args.batch
n_cores = args.n_cores


r_path = "results/snli/roberta_base/syn_p_h/batch{}".format(b)
rr_path = "raw_results/snli/roberta_base/syn_p_h/batch{}".format(b)

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

with open("roberta_script.sh", "w") as file:

    for rho in all_rhos:
        dgp = np.random.choice(dgp_seeds)
        train = np.random.choice(train_seeds)
        boot = np.random.choice(boot_seeds)

        command = "python3 wordnet_syn_test_roberta_base.py {:.1f} 687 {} {} {} {}\n".format(rho,
                                                                                   dgp,
                                                                                   train,
                                                                                   boot,
                                                                                   n_cores)
        file.write(command)
    file.write(
        "mv results/snli/roberta_base/syn_p_h/rho_* results/snli/roberta_base/syn_p_h/batch{}\n".format(b))
    file.write(
        "mv raw_results/snli/roberta_base/syn_p_h/rho_* raw_results/snli/roberta_base/syn_p_h/batch{}".format(b))
