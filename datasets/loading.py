"""Dataset loading."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
]

HYPBC_DATASETS = [
    "bc"
]

def load_data(dataset, normalize=True, data_size = None):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param data_size: number of samples in dataset
    @type data_size: int
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset,data_size)
    elif dataset in HYPBC_DATASETS:
        x,y = load_hypbc()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x0 = x[None, :, :]
    x1 = x[:, None, :]
    cos = (x0 * x1).sum(-1)
    similarities = 0.5 * (1 + cos)
    similarities = np.triu(similarities) + np.triu(similarities).T
    similarities[np.diag_indices_from(similarities)] = 1.0
    similarities[similarities > 1.0] = 1.0
    ####DEBUG####
    matrix_histogram(similarities)

    return x, y, similarities


def load_uci_data(dataset,data_size = None):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    i = 0
    with open(data_path, 'r') as f:
        for line in f:
            i = i + 1
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
            if data_size is not None and i == data_size:
                break
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    epslion = 1e-2
    std = x.std(0)
    x = (x - mean) / (std + epslion)
    return x, y

def load_hypbc_data(type="all",normalize = "none",visualize=False):
    if type == "all":
        data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/data_mrna_all.txt")
    elif type == "partial":
        data_path = os.path.join(os.environ["DATAPATH"], "breast_cancer/data_mrna.txt")
    else:
        print("bad type, use: \"partial\" or \"all\"")
        return
    clinical_data_path = os.path.join(os.environ["DATAPATH"],"breast_cancer/clinical_data.txt")

    #read data csv
    df = pd.read_csv(data_path, sep="\t", header=0)
    df.dropna(axis=0, inplace=True, how="any")
    df.drop(axis=1, inplace=True, labels="Entrez_Gene_Id")

    #sort features by variance
    df["variance"] = df.var(axis=1, numeric_only=True)
    df.sort_values(by=["variance"], ascending=False, inplace=True)

    if visualize:
        #add more information

        df["min"] = df.min(axis=1, numeric_only=True)
        df["max"] = df.max(axis=1, numeric_only=True)
        df["mean"] = df.mean(axis=1, numeric_only=True)


        #see some samples and shape
        print(df.shape)
        print(df.loc[0:5, :])

        #plot
        fig, axes = plt.subplots(nrows=1, ncols=5)
        fig.suptitle("5 most variant genes")
        fig.tight_layout()
        for i in range(5):
            cur_frame = df.iloc[i,:]
            var = cur_frame["variance"]
            mean = cur_frame["mean"]
            cur_frame = cur_frame.drop(labels=["Hugo_Symbol","min","max","mean","variance"])
            axes[i].hist(cur_frame, bins=50)
            axes[i].set_title('mean,var=({:.2f},{:.2f})'.format(mean, var),fontsize=8)

        df.drop(labels=["min","max","mean"],inplace=True,axis=1)


    #load clinical data
    clinical_df = pd.read_csv(clinical_data_path, sep="\t", header=4)

    # add the clinical data to the samples
    transposed_df = df.transpose()
    transposed_df.reset_index(inplace=True)
    transposed_df.rename(columns={"index": "SAMPLE_ID"}, inplace=True)
    df_full_data = pd.merge(transposed_df, clinical_df, on="SAMPLE_ID") #df_full: rows are patients, colums are features + clinical data

    #generate labels
    labels = df_full_data["CANCER_TYPE_DETAILED"]
    label_dict = {}
    for i,val in enumerate(labels.unique().tolist()):
        label_dict[val] = i

    labels.replace(label_dict, inplace=True)

    #generate the data
    hugo_sym = df["Hugo_Symbol"].to_list()
    df.drop(labels=['Hugo_Symbol','variance'], inplace=True,axis=1)
    sample_names = df.columns.values

    return df.to_numpy().transpose(), labels.to_numpy(), hugo_sym, sample_names

def generate_similarity_matrix(data, method = 'euclidean', features_dim = 1000,visualize=False):
    #assume data is a dataframe where each row is a sample and each column  is a feature
    filt_data = data[:,:features_dim]
    dist = sp.distance.cdist(filt_data,filt_data,method)

    #normalize similarity to [0,1]
    if method == 'cosine':
        mat = 0.5*(2-dist) # note that dist = 1-cos(t) => mat = 0.5(1 + cos(t))
    else:
        mat = 1-dist*(1/(np.amax(dist)))

    if visualize:
        v = plt.imshow(mat,cmap="gray")
        plt.title(f"similarity matrix, method:{method},#dim={features_dim}")
        plt.show()

    return mat
def load_hypbc(type="partial",
               normalize = "none",
               num_data_samples = -1,
               feature_dim = 50,
               method = "cosine",
               visualize=False):
    data, labels, feat_names, samp_names = load_hypbc_data(type=type,
                                                           normalize=normalize,
                                                           visualize=visualize)
    if num_data_samples > 2:
        data = data[:num_data_samples]
        labels = labels[:num_data_samples]

    sim_mat = generate_similarity_matrix(data,
                                         features_dim=feature_dim,
                                         method=method,
                                         visualize=visualize)
    ####DEBUG####
    matrix_histogram(sim_mat)
    return data, labels, sim_mat



def matrix_histogram(matrix):
    vector = matrix.flatten()
    #hist, bin_edges = np.histogram(vector)
    plt.figure()
    _ = plt.hist(vector,bins='auto')
    plt.show()

if __name__ == "__main__":
    load_hypbc()

