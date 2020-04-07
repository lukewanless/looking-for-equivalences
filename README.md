# logical-robustness
logical robustness


## Procedure

1) search hyperparams for xgb in the snli dataset

`python search_xgb.py 455 10 800 16`

2) transform train, dev and test datasets snli

`python3 syn_creation.py snli 16`

3) run test for xgb

`python wordnet_syn_test_xgb.py 0.0 455 342 3456 123 8`
