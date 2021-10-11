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

#model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_cityblock_full/model_sd1234_epch15.pkl"
#save_path = "tmp/best_cityblock.png"
#feature_dim = 98



model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_mahalanobis_full/model_sd1234_epch15.pkl"
save_path = "tmp/best_mahalanobis.png"
feature_dim = 181



#model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_cosine_full/model_sd1234_epch57.pkl"
#save_path = "tmp/best_cosine.png"
#feature_dim = 103



#model_path = "/home/raveh.be@staff.technion.ac.il/PycharmProjects/GED/embeddings/breast_cancer/best_euclidean_full/model_sd1234_epch9.pkl"
#save_path = "tmp/best_euclidean.png"
#feature_dim = 135



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

#########
def calculate_misclassification(model_path, save_path, feature_dim):
    method = 'euclidean' #doesn't really matter
    ##########
    model = None #load model
    with open(model_path,'rb') as f:
        #model = torch.load(f)
        model = HypHC(n_nodes=num_data_samples)
        model.load_state_dict(torch.load(f))
    tree = model.decode_tree(fast_decoding=True) # tree is digraph


    #first n in tree are the leaves, ordered in the same way as the labels
    #because we load the data the same way as in the training
    x, y_true, similarities, label_dict = load_hypbc_multi_group(num_groups=1,
                                                                     num_data_samples=num_data_samples,
                                                                     feature_dim=feature_dim,
                                                                     method=method,
                                                                     feature_correlation_thresh=feature_correlation_thresh,
                                                                     visualize=False)

    """fake tree for debugging"""
    #tree = nx.DiGraph()
    #tree.add_nodes_from([0,1,2,3,4])
    #tree.add_edges_from([(3,0),(3,1),(4,3),(4,2)])
    #y_true[0] = [3,3,3]

    # initial stats
    n = len(y_true[0])
    histograms = np.zeros((2*n-1, len(label_dict)))

    #fill histograms for leaves
    for leaf, label in enumerate(y_true[0]):
        histograms[leaf][label] = 1

    def get_histogram(node):
        sons = tree.successors(node)

        for son in sons: #if entered this loop, node is parent
            histograms[node] += get_histogram(son)

        return histograms[node]

    def get_label(node, count_type='normal'):
        if any(histograms[node] > 0): #then we already have a histogram
            if count_type== 'normal':
                return np.argmax(histograms[node])
            elif count_type== 'bas_her2':
                new_hist = np.zeros(len(label_dict)+1)
                basal_idx = label_dict.index('Basal')
                new_hist[basal_idx] = histograms[node,basal_idx]
                her2_idx = label_dict.index('Her2')
                new_hist[her2_idx] = histograms[node,her2_idx]
                new_hist[-1] = sum(histograms[node]) - her2_cnt - basal_cnt #index is a label that doesn't exist, so this is a mismatch
                return np.argmax(new_hist)
        else: #we dont have histogram, need to find it
            histograms[node] = get_histogram(node)
            return np.argmax(histograms[node])

    def get_brother(node):
        father = tree.predecessors(node).__next__() #get the father node number
        children = tree.successors(father)

        for child in children:
            if child is not node:
                return child

        assert False

    basal_idx = label_dict.index('Basal')
    her2_idx = label_dict.index('Her2')

    #calculate the classification rates
    mismatch_counter = {}
    match_counter = {}

    for stats_type in ['normal', 'bas_her2']:
        mismatch_counter[stats_type] = 0.0
        match_counter[stats_type] = 0.0
        #bas_her2_mismatch = 0.0
        #bas_her2_match_counter = 0.0

        for node in tree:
            if node >= n: #then this node is not a leaf. no need to calculate its familiarity
                continue
            cur_node_label = get_label(node, count_type=stats_type)
            if stats_type=='bas_her2' and cur_node_label not in [basal_idx,her2_idx]: #dont count stats for nodes not bas or her2 on this mode
                continue
            brother = get_brother(node)
            brother_node_label = get_label(brother,count_type=stats_type)
            if  cur_node_label != brother_node_label: #different label
                #if (label_dict[cur_node_label] == 'Basal' and label_dict[brother_node_label] == 'Her2') or \
                #   (label_dict[cur_node_label] == 'Her2' and label_dict[brother_node_label] == 'Basal'):
                #    bas_her2_mismatch += 1.0
                mismatch_counter[stats_type] += 1.0
            else: #same label
                match_counter[stats_type] += 1.0
                #if label_dict[cur_node_label] == 'Basal' or label_dict[cur_node_label] == 'Her2':
                #    bas_her2_match_counter += 1.0

    #normal
    misclassification_rate = mismatch_counter['normal']/n
    hit_classification_rate = match_counter['normal']/n
    print(f'general mis-classification_rate is : {misclassification_rate}'
          f'hit rate is: {hit_classification_rate}')
    #bas vs her2
    uniques ,counts = np.unique(y_true[0],return_counts=True)
    basal_cnt = counts[np.where(uniques == basal_idx)[0][0]] if np.where(uniques == basal_idx)[0].size>0 else 0
    her2_cnt = counts[np.where(uniques == her2_idx)[0][0]]   if np.where(uniques == her2_idx)[0].size>0 else 0
    n_bas_her2 =  basal_cnt + her2_cnt
    bas_her2_mismatch_rate = mismatch_counter['bas_her2']/n_bas_her2
    bas_her2_match_rate = match_counter['bas_her2']/n_bas_her2
    print(f'basal vs Her2 mis-clasification rate is: {bas_her2_mismatch_rate}'
          f'hit rate: {bas_her2_match_rate}')






#calculate_misclassification(model_path,save_path,feature_dim)

### visualize tree

method = 'euclidean' #doesn't really matter
##########
model = None #load model
with open(model_path,'rb') as f:
    #model = torch.load(f)
    model = HypHC(n_nodes=num_data_samples)
    model.load_state_dict(torch.load(f))
tree = model.decode_tree(fast_decoding=True) # tree is digraph


#first n in tree are the leaves, ordered in the same way as the labels
#because we load the data the same way as in the training
x, y_true, similarities, label_dict = load_hypbc_multi_group(num_groups=1,
                                                                 num_data_samples=num_data_samples,
                                                                 feature_dim=feature_dim,
                                                                 method=method,
                                                                 feature_correlation_thresh=feature_correlation_thresh,
                                                                 visualize=False)



visualize_tree(model,tree,y_true[0], save_path,label_dict)
