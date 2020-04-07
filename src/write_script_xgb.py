import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('batch',
                    type=int,
                    help='test batch')

args = parser.parse_args()

b = args.batch

r_path = "results/snli/xgb/syn_p_h/batch{}\n".format(b)
rr_path = "raw_results/snli/xgb/syn_p_h/batch{}\n".format(b)

if not os.path.exists(r_path):
    os.mkdir(r_path)
else:
    exit()

if not os.path.exists(rr_path):
    os.mkdir(rr_path)
else:
    exit()


all_rhos = np.array(range(0, 101)) / 100
dgp_seeds = range(1, 50000)
train_seeds = range(1, 50000)
boot_seeds = range(1, 50000)

assert os.path.exists(r_path)
assert len(os.listdir(r_path)) == 0
assert os.path.exists(rr_path)
assert len(os.listdir(rr_path)) == 0

with open("xgb_script.sh", "w") as file:

    for rho in all_rhos:
        dgp = np.random.choice(dgp_seeds)
        train = np.random.choice(train_seeds)
        boot = np.random.choice(boot_seeds)

        command = "python wordnet_syn_test_xgb.py {:.2f} 455 {} {} {} 16\n".format(rho,
                                                                                   dgp,
                                                                                   train,
                                                                                   boot)
        file.write(command)
    file.write(
        "mv results/snli/xgb/syn_p_h/rho_* results/snli/xgb/syn_p_h/batch{}\n".format(b))
    file.write(
        "mv raw_results/snli/xgb/syn_p_h/rho_* raw_results/snli/xgb/syn_p_h/batch{}".format(b))
