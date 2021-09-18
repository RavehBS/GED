"""Configuration parameters."""
from math import comb

config_args = {
    # training
    "seed": 1234,
    "epochs": 16,
    "batch_size": 128, # 64-512
    "learning_rate": 0.007420994156304837, #logscale 1e-5 - 1e0
    "eval_every": 1,
    "patience": 20,
    "optimizer": "RAdam",
    "save": 1,
    "fast_decoding": 1,
    "num_samples": -1, #comb(1000,2),

    # model
    "dtype": "double",
    "rank": 2, #we're working on 2d poincare disc, therfore rank=2
    "temperature": 0.11609589935073829, # 0.01 - 0.??
    "init_size": 0.08930598301183405, # 0.01-0.1
    "anneal_every": 34, # 10-100
    "anneal_factor": 0.8834221822915997, # 0.1-0.8
    "max_scale": 1 - 1e-3, # 0.9-(1-e-5)

    # dataset
    "dataset": "breast_cancer",
    #"dataset": "zoo",
    "similarity_metric": "cityblock", #'cosine','euclidean','mahalanobis','cityblock'
    "num_data_samples": 350, #any value below 2 will take all samples.
    "feature_dim": 98, #10-20e3 logscale
    "feature_correlation_thresh":0.9 #features that have a correlation of more than this value will be tossed
}



label_colors = {'0': 'r', '1': 'g', '2': 'b', '3': 'm','4' : 'y'}