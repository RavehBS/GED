import torch
import numpy as np
import networkx as nx

from datasets.loading import load_hypbc_multi_group
from model.hyphc import HypHC
from utils.metrics import dasgupta_cost
from visualize import visualize_tree
import os

"""
This module recieves an existing embedding, decodes it and calculates
evaluation metrics such as: Dasgupta cost, familiarity(?).
"""


os.makedirs("tmp", exist_ok=True)
"""
model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_cityblock_full/model_sd1234_epch15.pkl"
save_path = "tmp/best_cityblock.png"
feature_dim = 98
"""

"""
model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_mahalanobis_full/model_sd1234_epch15.pkl"
save_path = "tmp/best_mahalanobis.png"
feature_dim = 181
"""

model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_cosine_full/model_sd1234_epch57.pkl"
save_path = "tmp/best_cosine.png"
feature_dim = 103


"""
model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_euclidean_full/model_sd1234_epch9.pkl"
save_path = "tmp/best_euclidean.png"
feature_dim = 135
"""


num_data_samples = 1699

feature_correlation_thresh = 0.9

methods = ['cosine','euclidean','mahalanobis','cityblock']
####


def calculate_dasgupta_cost(model_path, save_path, feature_dim):
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
    return cost

####
tree = nx.DiGraph()
#leaves
tree.add_node('b',label=0)
tree.add_node('c',label=1)
tree.add_node('d',label=1)
#parents
tree.add_node('root',label=None)
tree.add_node('a',label=None)
#edges
tree.add_edge('root','a')
tree.add_edge('root','b')
tree.add_edge('a','c')
tree.add_edge('a','d')


# Pseudo code
"""
get_histogram(node):
    sons = get_succesors(node)
    if sons == None:
        #is leaf
        node.histogram[node,label] = 1
    else:
        #is parent
        node.histogram = get_histogram(sons[0]) + get_histogram(sons[1])

    return node.histogram

get_label(node):
    if node.histogram != None:
        return max(node.histogram)
    else: #we dont have histogram, need to find it
        node.histogram = get_histogram(node)
        return max(node.histogram)



for each leaf:
    brother = find_brother()
    if brother.label == None:
        brother.label = get_label(brother)

    if leaf.label != brother.label:
        cnt += 1
    else:
        continue

"""
