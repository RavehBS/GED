import torch
import numpy as np

from datasets.loading import load_hypbc_multi_group
from model.hyphc import HypHC
from utils.metrics import dasgupta_cost
from visualize import visualize_tree
import os

os.makedirs("tmp", exist_ok=True)
"""
model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_cityblock_full/model_sd1234_epch15.pkl"
save_path = "tmp/best_cityblock.png"
feature_dim = 98
"""
model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_mahalanobis_full/model_sd1234_epch15.pkl"
save_path = "tmp/best_mahalanobis.png"
feature_dim = 181


"""
model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_euclidean_full/model_sd1234_epch9.pkl"
save_path = "tmp/best_euclidean.png"
feature_dim = 135
"""


num_data_samples = 1699

feature_correlation_thresh = 0.9

methods = ['cosine','euclidean','mahalanobis','cityblock']
####



model = None
with open(model_path,'rb') as f:
    #model = torch.load(f)
    model = HypHC(n_nodes=num_data_samples)
    model.load_state_dict(torch.load(f))
tree = model.decode_tree(fast_decoding=True) # tree is digraph

x = None
y_true_prev = None
y_true = None
different_similarities = []
for method in methods:
    x, y_true, similarities, label_dict = load_hypbc_multi_group(num_groups=1,
                                                             num_data_samples=num_data_samples,
                                                             feature_dim=feature_dim,
                                                             method=method,
                                                             feature_correlation_thresh=feature_correlation_thresh,
                                                                 visualize=False)

    if y_true_prev is not None:
        assert(np.all(y_true_prev == y_true[0]))
    y_true_prev = y_true[0]
    different_similarities.append(similarities[0])

averaged_similarities = np.mean(different_similarities, axis=0)
cost = dasgupta_cost(tree, averaged_similarities)
visualize_tree(model,tree,y_true[0], save_path,label_dict)
print(f"Dasgupta cost: {cost}")
