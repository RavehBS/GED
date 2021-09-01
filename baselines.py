import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from datasets.loading import load_hypbc_multi_group
from time import gmtime, strftime
import matplotlib


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    matplotlib.rcParams['lines.linewidth'] = 0.5

    dendrogram(linkage_matrix,link_color_func =lambda x: 'k',**kwargs)

    #color the nodes
    label_colors = {'0': 'r', '1': 'g', '2': 'b', '3': 'm','4' : 'c'}

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(label_colors[lbl.get_text()])

    plt.show()

res_dir = 'baseline_results'
os.makedirs(res_dir,exist_ok=True) #save results here

linkages = {"SL" : 'single',
            "AL" : 'average',
            "CL" : 'complete',
            "WL" : 'ward'}


X, y_true, similarities, label_dict = load_hypbc_multi_group(num_groups=1,
                                                             num_data_samples=-1,
                                                             feature_dim=50,
                                                             method="cosine",
                                                             feature_correlation_thresh=0.9,
                                                             visualize=False)
for alg_type,param in linkages.items():

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=None,
                                    n_clusters=5,
                                    affinity='cosine',
                                    linkage=param,
                                    compute_distances=True)

    model = model.fit(X[0])
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model,labels = y_true[0])
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    plt_name = alg_type + "_" + strftime("%Y%m%d_%H%M", gmtime())
    plt.savefig(os.path.join(res_dir,plt_name))