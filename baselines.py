import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram
from datasets.loading import load_hypbc_multi_group
from time import gmtime, strftime
import matplotlib
from scipy.cluster.hierarchy import linkage
import networkx as nx
from utils.metrics import dasgupta_cost



from config import label_colors

def graph_from_linkage_mat(linkage_mat):
    n = linkage_mat.shape[0]+1

    g = nx.DiGraph()
    for i in range(linkage_mat.shape[0]):
        g.add_edge(int(n+i),int(linkage_mat[i,0]))
        g.add_edge(int(n+i),int(linkage_mat[i,1]))

    #testing:
    #plt.figure()
    #nx.draw_networkx(g)
    #plt.show()


    return g



def plot_dendrogram(linkage_matrix,similarity_mat, **kwargs):
    # Plot the corresponding dendrogram
    matplotlib.rcParams['lines.linewidth'] = 0.5
    fig = plt.figure()
    dnd=dendrogram(linkage_matrix,link_color_func =lambda x: 'k',**kwargs)


    #calculate dasgupta cost

    g = graph_from_linkage_mat(linkage_matrix)
    cost = dasgupta_cost(g,similarity_mat)

    #color the nodes
    ax = fig.gca()
    xlbls = ax.get_xmajorticklabels()
    fig.suptitle(f'HC by {alg_type}\nDusgupta cost={cost}')
    for lbl in xlbls:
        lbl.set_color(label_colors[lbl.get_text()])

    plt.show()

res_dir = 'baseline_results'
os.makedirs(res_dir,exist_ok=True) #save results here

linkages = {"WL" : ['ward','euclidean'],
            "SL" : ['single','cosine'],
            "AL" : ['average','cosine'],
            "CL" : ['complete','cosine']
            }


X, y_true, similarities, label_dict = load_hypbc_multi_group(num_groups=100,
                                                             num_data_samples=-1,
                                                             feature_dim=50,
                                                             method="cosine",
                                                             feature_correlation_thresh=0.9,                                                             visualize=False)
for alg_type,param in linkages.items():

    Z = linkage(X[0],method = param[0],metric = param[1])

    # plot the top three levels of the dendrogram
    plot_dendrogram(Z,similarities[0],labels = y_true[0])
    plt.xlabel("Bar color represents actual label for each leaf")

    legend_handles = []
    for key, color in label_colors.items():
        legend_handles.append(mpatches.Patch(color=color, label=label_dict[int(key)]))

    plt.legend(handles=legend_handles, labels=label_dict, loc='best',
              ncol=3, fontsize='small')
    plt.show()

    plt_name = alg_type + "_" + strftime("%Y%m%d_%H%M", gmtime())
    plt.savefig(os.path.join(res_dir,plt_name))