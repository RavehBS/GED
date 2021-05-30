"""Configuration parameters."""

config_args = {
    # training
    "seed": 1234,
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "eval_every": 1,
    "patience": 20,
    "optimizer": "RAdam",
    "save": 1,
    "fast_decoding": 1,
    "num_samples": 1_000_000, #-1

    # model
    "dtype": "double",
    "rank": 2,
    "temperature": 0.01,
    "init_size": 1e-3,
    "anneal_every": 20,
    "anneal_factor": 1.0,
    "max_scale": 1 - 1e-3,

    # dataset
    "dataset": "breast_cancer",
    "similarity_metric": "cosine", #'cosine','euclidean','mahalanobis','cityblock'
    "feature_dim": 10
}
